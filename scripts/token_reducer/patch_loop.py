"""Closed-loop: minimal patch → apply (apply_diff) → verify → surgical retry (deterministic, no LLM)."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from .apply_diff_bridge import get_apply_diffs
from .compressor import compress_candidates
from .execution_policy import ExecutionPolicy
from .models import Candidate
from .repo_map import RepoMap
from .retrieval_plan import RetrievalPlan
from .retriever import fts_retrieve
from .verify_loop import run_verification

_SIMPLE_REPLACE = re.compile(
    r"(?:\b(?:replace|change|fix)\s+)([^\s'\"]+)\s+(?:to|with)\s+([^\s'\"]+)",
    re.I,
)

# Error analysis (Python / tooling heuristics)
_RE_FILE_LINE_PY = re.compile(r'File\s+["\']([^"\']+)["\']\s*,\s*line\s+(\d+)', re.I)
_RE_FILE_LINE_COL = re.compile(r'File\s+["\']([^"\']+)["\']\s*,\s*line\s+(\d+)\s*,', re.I)
_RE_BAD_PY = re.compile(r"^(.+\.py)\s*:\s*line\s*(\d+)", re.I | re.M)
_RE_IMPORT_MOD = re.compile(r"No module named\s+['\"]?([^'\"\\s]+)['\"]?", re.I)
_RE_NAME_ERR = re.compile(r"name\s+['\"]?(\w+)['\"]?\s+is not defined", re.I)
_RE_TYPE_ERR = re.compile(r"TypeError:\s*(.+)", re.I)
_RE_FAILED_TEST = re.compile(r"FAILED\s+[^\s]+\s*::\s*(\w+)", re.I)
_RE_TRACEBACK_FILE = re.compile(r'^\s*File\s+["\']([^"\']+)["\']', re.M)


def _norm_key(path: str) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return path.replace("\\", "/")


def _parse_targets_from_patch_text(patch_text: str) -> tuple[str, ...]:
    out: list[str] = []
    for line in patch_text.splitlines():
        line = line.strip()
        if line.upper().startswith("<<<< SEARCH"):
            rest = line[len("<<<< SEARCH") :].strip()
            if rest:
                out.append(rest)
    return tuple(out)


def analyze_failure(
    error_message: str,
    patch_result: PatchResult | None,
    *,
    apply_messages: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """
    Extract file / line / symbol / error_type / hint from verifier or apply output.

    Uses regex + small heuristics only (no AST).
    """
    parts: list[str] = [error_message or ""]
    if apply_messages:
        parts.extend(apply_messages)
    blob = "\n".join(p for p in parts if p).strip()
    if not blob:
        return {
            "file": None,
            "line": None,
            "symbol": None,
            "error_type": "unknown",
            "hint": "empty error",
        }

    file_guess: str | None = None
    line_guess: int | None = None
    symbol_guess: str | None = None
    error_type = "unknown"
    hint = "inspect logs"

    if patch_result and patch_result.target_files:
        try:
            file_guess = str(Path(patch_result.target_files[0]).resolve())
        except OSError:
            file_guess = patch_result.target_files[0]

    m = _RE_FILE_LINE_PY.search(blob) or _RE_FILE_LINE_COL.search(blob)
    if m:
        file_guess = str(Path(m.group(1)).expanduser().resolve()) if m.group(1) else file_guess
        try:
            line_guess = int(m.group(2))
        except ValueError:
            line_guess = None

    if not line_guess:
        m2 = _RE_BAD_PY.search(blob)
        if m2:
            file_guess = file_guess or str(Path(m2.group(1)).expanduser().resolve())
            try:
                line_guess = int(m2.group(2))
            except ValueError:
                pass

    low = blob.lower()
    if "syntaxerror" in low or "invalid syntax" in low or "unexpected eof" in low:
        error_type = "syntax"
        hint = "syntax issue in edited region; fix locally or adjust SEARCH/REPLACE anchor"
    elif "importerror" in low or "modulenotfounderror" in low or "no module named" in low:
        error_type = "import"
        hint = "missing import or wrong module path"
        im = _RE_IMPORT_MOD.search(blob)
        if im:
            symbol_guess = im.group(1)
    elif "typeerror" in low:
        error_type = "type"
        hint = "wrong type or signature mismatch"
        tm = _RE_TYPE_ERR.search(blob)
        if tm:
            symbol_guess = (tm.group(1) or "")[:120].strip()
    elif "nameerror" in low:
        error_type = "runtime"
        hint = "undefined name; check binding or import"
        nm = _RE_NAME_ERR.search(blob)
        if nm:
            symbol_guess = nm.group(1)
    elif "failed" in low and ("test" in low or "::" in blob or "pytest" in low):
        error_type = "test"
        hint = "test assertion or fixture failure"
        tf = _RE_FAILED_TEST.search(blob)
        if tf:
            symbol_guess = tf.group(1)
    elif "traceback" in low or "error" in low:
        error_type = "runtime"
        hint = "runtime failure; follow stack to callsite"
        tf = _RE_TRACEBACK_FILE.search(blob)
        if tf and not file_guess:
            try:
                file_guess = str(Path(tf.group(1)).expanduser().resolve())
            except OSError:
                file_guess = tf.group(1)
    elif "search text not found" in low or "not found" in low and "apply" in low:
        error_type = "apply"
        hint = "patch anchor mismatch; narrow SEARCH block to unique snippet"

    return {
        "file": file_guess,
        "line": line_guess,
        "symbol": symbol_guess,
        "error_type": error_type,
        "hint": hint,
    }


def build_retry_plan(
    failure_info: dict[str, Any],
    previous_plan: RetrievalPlan | None,
    policy: ExecutionPolicy,
) -> dict[str, Any]:
    """
    Surgical retry: only failing files, tight line window, slashed budget, no repo-wide pull.
    """
    _ = previous_plan
    et = str(failure_info.get("error_type") or "unknown")
    focus: list[str] = []
    if failure_info.get("file"):
        focus.append(str(failure_info["file"]))

    # 50–70% context reduction vs typical narrow pass
    word_scale = 0.32 if et == "syntax" else 0.38
    if isinstance(policy, ExecutionPolicy):
        word_scale = min(0.35, max(0.28, word_scale))

    plan: dict[str, Any] = {
        "focus_files": focus,
        "window_half_lines": 55 if et in ("syntax", "import") else 85,
        "fts_limit": 0 if et == "syntax" else (4 if et == "import" else 8),
        "skip_fts": et == "syntax",
        "include_callers": et in ("type", "runtime"),
        "include_callees": et in ("type", "runtime"),
        "max_chunks": 1 if et == "syntax" else 2,
        "word_budget_scale": word_scale,
        "disable_wide_retrieval": True,
        "disable_expansion": et != "runtime",
        "relevance_floor_bump": 0.08,
    }
    if et == "test":
        plan["window_half_lines"] = 100
        plan["fts_limit"] = 6
        plan["include_callers"] = True
        plan["max_chunks"] = 2
    return plan


def refine_patch(
    previous_patch: PatchResult,
    failure_info: dict[str, Any],
    context: str,
) -> PatchResult | None:
    """
    Adjust the previous patch minimally when rules allow; otherwise return None.

    Keeps working SEARCH/REPLACE blocks and appends a small fix block when safe.
    """
    if not previous_patch.applicable or not previous_patch.patch_text.strip():
        return None
    et = str(failure_info.get("error_type") or "")
    mod = failure_info.get("symbol")
    if et != "import" or not mod or not isinstance(mod, str):
        return None
    # Only add import if context shows file head and import absent
    head = (context or "").splitlines()[:40]
    head_txt = "\n".join(head)
    imp_line = f"import {mod.split('.')[0]}"
    if imp_line in head_txt or f"from {mod.split('.')[0]} import" in head_txt:
        return None
    rels = _parse_targets_from_patch_text(previous_patch.patch_text)
    if not rels:
        return None
    rel = rels[0]
    first = head[0] if head else ""
    if not first.strip():
        return None
    extra = (
        f"\n<<<< SEARCH {rel}\n{first}\n==== REPLACE\n{imp_line}\n{first}\n>>>>"
    )
    new_text = previous_patch.patch_text.rstrip() + extra
    return PatchResult(
        target_files=previous_patch.target_files,
        diff_hunks=new_text[:8000],
        rationale="refined: prepend missing import (minimal block)",
        patch_text=new_text,
        applicable=True,
    )


def _failure_actionable(failure_info: dict[str, Any], err_blob: str) -> bool:
    if not err_blob.strip():
        return False
    if failure_info.get("error_type") == "unknown" and not failure_info.get("file"):
        return False
    # Non-actionable noise
    if "no diff blocks" in err_blob.lower():
        return False
    return True


def _read_file_window(path: Path, line: int | None, half_lines: int) -> str:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    lines = raw.splitlines()
    if not lines:
        return ""
    if line is None or line < 1:
        j = min(len(lines), max(1, 2 * half_lines))
        return "\n".join(lines[:j])
    idx = line - 1
    i = max(0, idx - half_lines)
    j = min(len(lines), idx + half_lines)
    return "\n".join(lines[i:j])


def _surgical_candidates(
    conn: Any,
    workspace_root: Path,
    retry_plan: dict[str, Any],
    failure_info: dict[str, Any],
    query_hint: str,
) -> list[Candidate]:
    """Focused chunks only: one windowed file slice and optional tiny FTS."""
    out: list[Candidate] = []
    focus = retry_plan.get("focus_files") or []
    half = int(retry_plan.get("window_half_lines") or 60)
    fp = failure_info.get("file")
    line = failure_info.get("line")
    if isinstance(line, str) and line.isdigit():
        line = int(line)
    if not isinstance(line, int):
        line = None

    paths: list[Path] = []
    if fp:
        try:
            paths.append(Path(str(fp)).resolve())
        except OSError:
            pass
    for f in focus:
        try:
            p = Path(str(f)).resolve()
            if p not in paths:
                paths.append(p)
        except OSError:
            continue

    ws = workspace_root.resolve()
    for p in paths[:1]:
        if not p.is_file():
            continue
        try:
            p.relative_to(ws)
        except ValueError:
            continue
        window = _read_file_window(p, line, half)
        if not window.strip():
            continue
        cid = abs(hash(str(p))) % (10**9 - 1) or 1
        out.append(
            Candidate(
                chunk_id=cid,
                source=str(p),
                chunk_index=0,
                text=window,
                token_estimate=max(8, len(window) // 4),
                final_score=1.0,
            )
        )

    if retry_plan.get("skip_fts") or int(retry_plan.get("fts_limit") or 0) <= 0:
        return out[: int(retry_plan.get("max_chunks") or 2)]

    lim = max(1, min(12, int(retry_plan.get("fts_limit") or 6)))
    qtoks: list[str] = []
    if failure_info.get("symbol"):
        qtoks.append(str(failure_info["symbol"]))
    if failure_info.get("file"):
        qtoks.append(Path(str(failure_info["file"])).stem)
    q = " ".join(qtoks) if qtoks else query_hint[:120]
    hits = fts_retrieve(conn, q, limit=lim)
    keys = {_norm_key(str(f)) for f in focus if f}
    filtered = [h for h in hits if _norm_key(h.source) in keys] if keys else hits
    pick = filtered[: max(0, int(retry_plan.get("max_chunks") or 2) - len(out))]
    out.extend(pick)
    return out[: int(retry_plan.get("max_chunks") or 2)]


def _narrow_policy_surgical(
    policy: ExecutionPolicy,
    failing_files: list[str],
    retry_plan: dict[str, Any],
) -> ExecutionPolicy:
    if not isinstance(policy, ExecutionPolicy):
        return policy
    boost = frozenset(failing_files) | policy.boosted_sources
    scale = float(retry_plan.get("word_budget_scale") or 0.35)
    wb = max(48, int(policy.effective_word_budget * scale))
    rf = min(0.42, policy.effective_relevance_floor + float(retry_plan.get("relevance_floor_bump") or 0.06))
    inc_c = bool(retry_plan.get("include_callees"))
    inc_p = bool(retry_plan.get("include_callers"))
    return replace(
        policy,
        boosted_sources=boost,
        include_callers=inc_p,
        include_callees=inc_c,
        effective_word_budget=wb,
        effective_relevance_floor=rf,
    )


class PatchResult:
    __slots__ = ("target_files", "diff_hunks", "rationale", "patch_text", "applicable")

    def __init__(
        self,
        *,
        target_files: tuple[str, ...],
        diff_hunks: str,
        rationale: str,
        patch_text: str,
        applicable: bool,
    ) -> None:
        self.target_files = target_files
        self.diff_hunks = diff_hunks
        self.rationale = rationale
        self.patch_text = patch_text
        self.applicable = applicable


class ApplyOutcome:
    __slots__ = ("success", "messages", "modified_files")

    def __init__(
        self,
        success: bool,
        messages: tuple[str, ...],
        modified_files: tuple[str, ...],
    ) -> None:
        self.success = success
        self.messages = messages
        self.modified_files = modified_files


def _touched_paths_from_patch(patch: PatchResult, workspace_root: Path) -> list[Path]:
    paths: list[Path] = []
    ws = workspace_root.resolve()
    for rel in _parse_targets_from_patch_text(patch.patch_text):
        p = (ws / rel).resolve()
        if p.is_file():
            paths.append(p)
    for tf in patch.target_files:
        p = Path(tf)
        if p.is_file() and p.resolve() not in {x.resolve() for x in paths}:
            paths.append(p.resolve())
    return paths


def generate_patch(
    query: str,
    policy: ExecutionPolicy,
    plan: RetrievalPlan | None,
    selected: list[Candidate],
    repo_map: RepoMap | None,
    *,
    workspace_root: Path,
    patch_text_override: str | None = None,
) -> PatchResult:
    """
    Build a minimal SEARCH/REPLACE patch without an LLM.

    Heuristic: ``replace OLD with NEW`` / ``change OLD to NEW`` and OLD appears
    exactly once in the first on-disk candidate under ``workspace_root``.
    ``patch_text_override`` (tests) wins when set.
    """
    _ = plan, repo_map
    if patch_text_override:
        targets: list[str] = []
        for r in _parse_targets_from_patch_text(patch_text_override):
            try:
                targets.append(str((workspace_root / r).resolve()))
            except OSError:
                targets.append(r)
        return PatchResult(
            target_files=tuple(targets),
            diff_hunks=patch_text_override[:8000],
            rationale="patch_text_override",
            patch_text=patch_text_override,
            applicable=True,
        )
    if not policy.patch_first:
        return PatchResult(
            target_files=(),
            diff_hunks="",
            rationale="patch_first is false",
            patch_text="",
            applicable=False,
        )
    m = _SIMPLE_REPLACE.search(query or "")
    if not m or not selected:
        return PatchResult(
            target_files=(),
            diff_hunks="",
            rationale="no simple replace pattern or empty selection",
            patch_text="",
            applicable=False,
        )
    old_tok, new_tok = m.group(1), m.group(2)
    ws = workspace_root.resolve()
    for cand in selected[:8]:
        p = Path(cand.source)
        try:
            if not p.is_file():
                continue
            p.resolve().relative_to(ws)
        except ValueError:
            continue
        except OSError:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        if old_tok not in text or text.count(old_tok) != 1:
            continue
        try:
            rel = p.resolve().relative_to(ws).as_posix()
        except ValueError:
            rel = p.name
        block = f"<<<< SEARCH {rel}\n{old_tok}\n==== REPLACE\n{new_tok}\n>>>>"
        return PatchResult(
            target_files=(str(p.resolve()),),
            diff_hunks=block,
            rationale=f"single-token replace {old_tok!r} -> {new_tok!r} in {rel}",
            patch_text=block,
            applicable=True,
        )
    return PatchResult(
        target_files=(),
        diff_hunks="",
        rationale="no eligible on-disk file with unique token match",
        patch_text="",
        applicable=False,
    )


def apply_patch_result(
    patch: PatchResult,
    workspace_root: Path,
    *,
    dry_run: bool = False,
) -> ApplyOutcome:
    if not patch.applicable or not patch.patch_text.strip():
        return ApplyOutcome(False, ("no applicable patch",), ())
    apply_diffs = get_apply_diffs()
    res = apply_diffs(
        patch.patch_text,
        working_dir=workspace_root.resolve(),
        default_file=None,
        dry_run=dry_run,
    )
    msgs = tuple(str(x) for x in res.get("messages", []))
    applied = int(res.get("applied", 0) or 0)
    failed = int(res.get("failed", 0) or 0)
    ok = failed == 0 and applied > 0
    modified = tuple(patch.target_files) if ok else ()
    return ApplyOutcome(ok, msgs, modified)


def run_closed_edit_loop(
    state: Any,
    *,
    workspace_root: Path,
    dry_run: bool = False,
    patch_text_override: str | None = None,
) -> dict[str, Any]:
    """
    generate_patch → apply → verify → analyze → surgical retry (max 2).

    Restores query / policy / selected / bullets after the loop; details in
    ``state.agent_result`` including ``engineer_log`` for demo-style trace.
    """
    snap_query = state.query
    snap_policy = state.policy
    snap_selected = list(state.selected)
    snap_bullets = list(state.bullets)
    engineer_log: list[str] = []

    def _trace(msg: str) -> None:
        engineer_log.append(msg)

    if not snap_policy or not snap_policy.patch_first:
        out: dict[str, Any] = {
            "closed_loop": "skipped",
            "reason": "patch_first_false_or_no_policy",
            "plan": {"retrieval_plan": _plan_dict(state.retrieval_plan), "task_mode": None},
            "patch": None,
            "verification": {"status": "skipped", "retries": 0},
            "token_stats": {"used": 0, "saved": 0},
            "engineer_log": engineer_log,
        }
        state.agent_result = out
        return out

    vp = snap_policy.verification_plan or {}
    max_retries = min(2, int(vp.get("max_retries", 2)))

    query_eff = snap_query
    policy_eff = snap_policy
    selected_eff = snap_selected
    attempts: list[dict[str, Any]] = []

    final_patch: PatchResult | None = None
    retry_count = 0
    ver_status = "skipped"
    pending_retry_patch: PatchResult | None = None

    for attempt in range(max_retries + 1):
        if pending_retry_patch is not None:
            patch = pending_retry_patch
            pending_retry_patch = None
            _trace("Applying refined / carry-forward patch from prior failure analysis.")
        else:
            patch = generate_patch(
                query_eff,
                policy_eff,
                state.retrieval_plan,
                selected_eff,
                state.repo_map,
                workspace_root=workspace_root,
                patch_text_override=patch_text_override if attempt == 0 else None,
            )
        final_patch = patch
        step: dict[str, Any] = {
            "attempt": attempt,
            "query": query_eff[:500],
            "applicable": patch.applicable,
            "rationale": patch.rationale,
            "engineer_log_tail": list(engineer_log[-6:]),
        }

        if not patch.applicable:
            attempts.append(step)
            ver_status = "skipped"
            _trace("Patch not applicable; stopping (nothing to apply).")
            break

        if attempt == 0:
            _trace("Generated initial minimal patch.")

        apply_out = apply_patch_result(patch, workspace_root, dry_run=dry_run)
        step["apply_success"] = apply_out.success
        step["apply_messages"] = list(apply_out.messages)[:12]

        touched = _touched_paths_from_patch(patch, workspace_root)
        if not touched and patch.target_files:
            touched = [Path(f) for f in patch.target_files if Path(f).is_file()]

        ver = run_verification(
            touched,
            vp,
            cmds=state.runtime.shadow_linter_cmds,
            timeout=int(vp.get("lint_timeout_seconds", 45)),
        )
        step["verification_success"] = ver.success
        step["verification_errors"] = list(ver.errors)[:8]
        attempts.append(step)

        if apply_out.success and ver.success:
            ver_status = "passed"
            retry_count = attempt
            _trace("Verification passed.")
            break

        ver_status = "failed"
        retry_count = attempt
        err_blob = "; ".join(ver.errors) if ver.errors else "; ".join(apply_out.messages)
        _trace("Patch failed → analyzing error...")
        finfo = analyze_failure(err_blob, patch, apply_messages=apply_out.messages)
        step["failure_analysis"] = finfo

        if attempt >= max_retries:
            _trace("Max retries reached; not re-running pipeline.")
            break

        if not _failure_actionable(finfo, err_blob):
            _trace("Failure not actionable; no surgical retry.")
            break

        _trace(
            f"Detected {finfo.get('error_type')} in "
            f"{Path(str(finfo.get('file') or 'unknown')).name}: {finfo.get('hint')}"
        )

        retry_plan = build_retry_plan(finfo, state.retrieval_plan, snap_policy)
        step["retry_plan"] = retry_plan

        failing = list(ver.failing_files) or ([str(p) for p in touched] if touched else [])
        if finfo.get("file"):
            failing.append(str(finfo["file"]))
        failing = list(dict.fromkeys(failing))[:4]

        policy_eff = _narrow_policy_surgical(snap_policy, failing, retry_plan)
        if snap_policy.rewrite_query:
            query_eff = f"Fix ({finfo.get('error_type')}): {err_blob}"[:900]
        else:
            query_eff = f"{snap_query} ({finfo.get('error_type')}): {err_blob}"[:900]

        selected_eff = _surgical_candidates(
            state.conn,
            workspace_root,
            retry_plan,
            finfo,
            query_eff,
        )
        ctx_preview = "\n".join((c.text[:400] for c in selected_eff[:2]))
        step["surgical_context_preview"] = ctx_preview
        n_files = len({c.source for c in selected_eff})
        n_lines = sum(len(c.text.splitlines()) for c in selected_eff)
        _trace(f"Retrying with focused context ({n_files} file(s), ~{n_lines} lines, no wide pull).")

        refined = refine_patch(patch, finfo, "\n".join(c.text for c in selected_eff))
        if refined:
            pending_retry_patch = refined
            _trace("Prepared refined patch for next attempt.")
        else:
            _trace("No local patch refinement; next attempt will regenerate from focused context only.")

        state.policy = policy_eff
        state.bullets = compress_candidates(
            query_eff,
            selected_eff,
            word_budget=policy_eff.effective_word_budget,
            relevance_floor=policy_eff.effective_relevance_floor,
            code_invariant_level=policy_eff.code_invariant_level,
            must_keep_tokens=policy_eff.must_keep_symbol_tokens,
        )

    state.query = snap_query
    state.policy = snap_policy
    state.selected = snap_selected
    state.bullets = snap_bullets

    token_used = sum(len(b.split()) for b in snap_bullets)
    token_saved = max(0, sum(c.token_estimate for c in snap_selected) - token_used)

    if ver_status == "passed":
        closed = "completed"
    elif ver_status == "failed":
        closed = "failed"
    elif ver_status == "skipped":
        closed = "skipped"
    else:
        closed = "partial"

    out = {
        "closed_loop": closed,
        "plan": {
            "task_mode": snap_policy.task_mode,
            "rerank_strategy": snap_policy.rerank_strategy,
            "retrieval_plan": _plan_dict(state.retrieval_plan),
            "retry_on_low_score": snap_policy.retry_on_low_score,
            "rewrite_query": snap_policy.rewrite_query,
        },
        "patch": None
        if final_patch is None or not final_patch.applicable
        else {
            "target_files": list(final_patch.target_files),
            "diff_hunks": final_patch.diff_hunks[:12000],
            "rationale": final_patch.rationale,
        },
        "verification": {"status": ver_status, "retries": retry_count},
        "attempts": attempts,
        "token_stats": {"used": token_used, "saved": token_saved},
        "engineer_log": engineer_log,
    }
    state.agent_result = out
    return out


def _plan_dict(plan: RetrievalPlan | None) -> dict[str, Any]:
    if plan is None:
        return {}
    return {
        "fts_cap": plan.fts_cap,
        "effective_top_k": plan.effective_top_k,
        "expand_neighbor_chunks": plan.expand_neighbor_chunks,
        "second_pass_on_weak_pool": plan.second_pass_on_weak_pool,
    }
