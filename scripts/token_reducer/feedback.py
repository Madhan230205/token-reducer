"""Lightweight local logging of pipeline outcomes (no ML, no training)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model_tuning import compression_model_tuning, model_scale


def _default_log_path() -> Path:
    root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if root:
        return Path(root) / ".cache" / "token-reducer" / "feedback.jsonl"
    return Path.home() / ".cache" / "token-reducer" / "feedback.jsonl"


def log_result(
    prompt: str,
    context: str,
    response: str | None = None,
    *,
    extra: dict[str, Any] | None = None,
    log_path: Path | None = None,
) -> None:
    """Append one JSON line: context size, chunk hints, optional quality signal.

    Use ``extra["outcome"]`` / ``extra["pipeline_outcome"]`` with ``"ok"`` / ``"bad"`` so
    :func:`feedback_loop_adjustments` can tighten future runs.
    """
    path = log_path or _default_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": int(time.time()),
            "prompt_chars": len(prompt or ""),
            "context_chars": len(context or ""),
            "context_words": len((context or "").split()),
            "response_chars": len(response or "") if response is not None else None,
            "extra": extra or {},
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        return


def median_recent_context_words(
    log_path: Path | None = None,
    *,
    tail_lines: int = 48,
) -> int | None:
    """Median ``context_words`` from recent JSONL rows — guides lean routing when outputs run fat."""
    path = log_path or _default_log_path()
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    words_list: list[int] = []
    for line in raw[-tail_lines:]:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        w = obj.get("context_words")
        if isinstance(w, int) and w >= 0:
            words_list.append(w)
    if len(words_list) < 4:
        return None
    words_list.sort()
    return words_list[len(words_list) // 2]


@dataclass(frozen=True)
class FeedbackLoopAdjustments:
    """Cached per run: tightens or loosens pipeline from recent JSONL (no training)."""

    retrieval_scale_mult: float
    relevance_floor_delta: float


def build_adaptation_brief(
    *,
    adj: FeedbackLoopAdjustments,
    model: str | None,
    feedback_source_boosts: dict[str, float],
    subagent_meta: dict[str, Any] | None,
    compression_level_effective: str | None = None,
    execution_tier: str | None = None,
) -> list[str]:
    """Short human-readable lines: surfaces predictive + self-adjusting behavior (for payloads / debug).

    This narrates knobs the pipeline already applies — no extra ML.
    """
    lines: list[str] = []
    m = (model or "sonnet").strip().lower()
    scale = model_scale(model)
    tuning = compression_model_tuning(model)
    if "haiku" in m:
        lines.append(
            "Model-aware shaping: Haiku profile — tighter pack, stronger query anchoring, "
            f"higher relevance floor (+{tuning.relevance_floor_shift:.3f}), rank similarity ×{tuning.rank_sim_scale:.2f}."
        )
    elif "opus" in m:
        lines.append(
            "Model-aware shaping: Opus profile — slightly roomier budget factor "
            f"×{tuning.budget_factor:.3f}, softer relevance floor ({tuning.relevance_floor_shift:+.3f}), "
            f"rank similarity ×{tuning.rank_sim_scale:.2f}."
        )
    else:
        lines.append(
            f"Model-aware shaping: default profile (budget scale ×{scale:.2f} vs Sonnet baseline)."
        )

    if execution_tier:
        lines.append(f"Route tier `{execution_tier}` — retrieval/subagent/compression stages scaled to match.")

    if compression_level_effective and compression_level_effective.lower() not in ("medium", "standard"):
        lines.append(f"Compression aggressiveness this pass: `{compression_level_effective}`.")

    if adj.retrieval_scale_mult != 1.0 or adj.relevance_floor_delta != 0.0:
        lines.append(
            "Feedback loop from recent JSONL: retrieval scale "
            f"×{adj.retrieval_scale_mult:.2f}, relevance floor Δ{adj.relevance_floor_delta:+.3f} "
            "(tighter when past contexts/outcomes were fat or weak)."
        )

    if feedback_source_boosts:
        top = sorted(feedback_source_boosts.items(), key=lambda kv: -kv[1])[:5]
        hinted = ", ".join(k for k, _ in top)
        lines.append(
            f"Predictive retrieval bias: prior fat contexts favored these sources (+score): {hinted}."
        )

    if isinstance(subagent_meta, dict):
        saved = subagent_meta.get("tokens_saved")
        if isinstance(saved, int) and saved > 0:
            lines.append(f"Deterministic subagent passes estimated ~{saved} fewer tokens before compression.")
        spec = subagent_meta.get("specialization")
        if isinstance(spec, dict):
            bits: list[str] = []
            if spec.get("bug_fix_error_terms_prefixed"):
                bits.append("exception-type prefetch in ranking")
            if spec.get("navigation_precision"):
                bits.append("navigation-stable ordering (variance off)")
            intent = spec.get("legacy_intent")
            if intent == "bug_fix" and not bits:
                bits.append("bug-fix coordination steps")
            if bits:
                lines.append("Specialized chain cues: " + "; ".join(bits) + ".")

    return lines


def feedback_loop_adjustments(
    log_path: Path | None = None,
    *,
    tail_lines: int = 56,
    min_samples: int = 6,
    fat_median_words: int = 165,
    lean_median_words: int = 58,
) -> FeedbackLoopAdjustments:
    """If recent packets were too fat, retrieve/compress harder next time (and vice versa).

    Hosts may set ``extra.outcome`` / ``extra.pipeline_outcome`` to ``bad`` | ``ok`` to accelerate tightening.
    """
    path = log_path or _default_log_path()
    neutral = FeedbackLoopAdjustments(1.0, 0.0)
    if not path.is_file():
        return neutral
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return neutral
    words_list: list[int] = []
    bad = 0
    labelled = 0
    for line in raw[-tail_lines:]:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        w = obj.get("context_words")
        if isinstance(w, int) and w >= 0:
            words_list.append(w)
        extra = obj.get("extra")
        if isinstance(extra, dict):
            o = extra.get("outcome") or extra.get("pipeline_outcome")
            if o in ("bad", "negative", "fail"):
                bad += 1
                labelled += 1
            elif o in ("ok", "good", "success"):
                labelled += 1

    if len(words_list) < min_samples:
        return neutral

    words_list.sort()
    med = words_list[len(words_list) // 2]

    r_mult = 1.0
    rf_delta = 0.0
    if med >= fat_median_words:
        r_mult = 0.89
        rf_delta = 0.028
    elif med <= lean_median_words:
        r_mult = 1.05
        rf_delta = -0.014

    if labelled >= 5 and bad / labelled > 0.45:
        r_mult *= 0.93
        rf_delta += 0.018

    r_mult = max(0.84, min(1.09, r_mult))
    rf_delta = max(-0.022, min(0.045, rf_delta))
    return FeedbackLoopAdjustments(r_mult, rf_delta)


def read_feedback_source_boosts(
    log_path: Path | None = None,
    *,
    tail_lines: int = 72,
    min_words: int = 24,
    max_boost: float = 0.06,
) -> dict[str, float]:
    """Lightweight ranking hints from recent JSONL: boost sources seen in 'fat' outcomes.

    Reads the last ``tail_lines`` of the log; rows with ``context_words`` above ``min_words``
    contribute ``extra.selected_sources`` path basenames with a capped positive delta.
    """
    path = log_path or _default_log_path()
    if not path.is_file():
        return {}
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    rows = raw[-tail_lines:]
    counts: dict[str, int] = {}
    for line in rows:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        words = int(obj.get("context_words") or 0)
        if words < min_words:
            continue
        extra = obj.get("extra")
        if not isinstance(extra, dict):
            continue
        srcs = extra.get("selected_sources")
        if not isinstance(srcs, list):
            continue
        for s in srcs:
            if not isinstance(s, str) or not s.strip():
                continue
            key = Path(s).name
            counts[key] = counts.get(key, 0) + 1
    out: dict[str, float] = {}
    for name, n in counts.items():
        out[name] = min(max_boost, 0.012 * (n**0.5))
    return out


def _strategy_prune_deltas_from_jsonl(
    path: Path,
    *,
    tail_lines: int = 200,
    min_samples: int = 4,
    fat_words: int = 200,
    lean_words: int = 72,
) -> dict[str, int]:
    if not path.is_file():
        return {}
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    rows = raw[-tail_lines:]
    buckets: dict[str, list[int]] = {}
    for line in rows:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        extra = obj.get("extra")
        if not isinstance(extra, dict):
            continue
        sid = extra.get("strategy_id")
        if not isinstance(sid, str) or not sid.strip():
            continue
        words = int(obj.get("context_words") or 0)
        buckets.setdefault(sid.strip(), []).append(words)
    out: dict[str, int] = {}
    for sid, ws in buckets.items():
        if len(ws) < min_samples:
            continue
        med = sorted(ws)[len(ws) // 2]
        if med >= fat_words:
            out[sid] = -1
        elif med <= lean_words:
            out[sid] = 1
        else:
            out[sid] = 0
    return out


def strategy_prune_deltas_from_feedback_log(
    log_path: Path | None = None,
    *,
    tail_lines: int = 200,
    min_samples: int = 4,
    fat_words: int = 200,
    lean_words: int = 72,
) -> dict[str, int]:
    """Snapshot prune deltas from the feedback JSONL only (for persistence hooks)."""
    path = log_path or _default_log_path()
    return _strategy_prune_deltas_from_jsonl(
        path,
        tail_lines=tail_lines,
        min_samples=min_samples,
        fat_words=fat_words,
        lean_words=lean_words,
    )


def _workspace_prune_ema_path(workspace_root: Path) -> Path:
    return workspace_root / ".token-reducer" / "prune_ema.json"


def load_workspace_prune_ema(workspace_root: Path | None) -> dict[str, float]:
    if workspace_root is None:
        return {}
    path = _workspace_prune_ema_path(workspace_root)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    ema = data.get("ema")
    if not isinstance(ema, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in ema.items():
        if isinstance(k, str) and isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def persist_workspace_prune_ema(workspace_root: Path | None, log_deltas: dict[str, int]) -> None:
    """Smooth logged prune signals into a per-workspace file (cross-session)."""
    if workspace_root is None or not log_deltas:
        return
    path = _workspace_prune_ema_path(workspace_root)
    prev = load_workspace_prune_ema(workspace_root)
    ema: dict[str, float] = dict(prev)
    for sid, d in log_deltas.items():
        ema[sid] = 0.82 * ema.get(sid, 0.0) + 0.18 * float(int(d))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ema": ema}, indent=2), encoding="utf-8")
    except OSError:
        return


def read_strategy_prune_adjustments(
    log_path: Path | None = None,
    *,
    tail_lines: int = 200,
    min_samples: int = 4,
    fat_words: int = 200,
    lean_words: int = 72,
    workspace_root: Path | None = None,
) -> dict[str, int]:
    """From feedback JSONL plus optional workspace EMA, suggest small ``prune_k`` deltas per strategy."""
    path = log_path or _default_log_path()
    log_d = _strategy_prune_deltas_from_jsonl(
        path,
        tail_lines=tail_lines,
        min_samples=min_samples,
        fat_words=fat_words,
        lean_words=lean_words,
    )
    if workspace_root is None:
        return log_d
    persisted = load_workspace_prune_ema(workspace_root)
    keys = set(log_d) | set(persisted)
    merged: dict[str, int] = {}
    for sid in keys:
        v = 0.55 * float(log_d.get(sid, 0)) + 0.45 * float(persisted.get(sid, 0.0))
        merged[sid] = max(-1, min(1, int(round(v))))
    return merged


def feedback_loop_adjustments_with_adaptive(
    log_path: Path | None = None,
    *,
    workspace_root: Path | None = None,
    tail_lines: int = 56,
    min_samples: int = 6,
    fat_median_words: int = 165,
    lean_median_words: int = 58,
) -> FeedbackLoopAdjustments:
    """Legacy JSONL adjustments layered with promoted adaptive committed deltas.

    Order: compute base from :func:`feedback_loop_adjustments`, then add adaptive
    ``retrieval_scale_mult_delta`` / ``relevance_floor_delta`` from disk (clamped).
    """
    base = feedback_loop_adjustments(
        log_path,
        tail_lines=tail_lines,
        min_samples=min_samples,
        fat_median_words=fat_median_words,
        lean_median_words=lean_median_words,
    )
    from .adaptive_feedback.constants import adaptive_disabled
    from .adaptive_feedback.state_store import load_committed

    if adaptive_disabled():
        return base
    committed = load_committed(workspace_root)
    if committed is None:
        return base
    r = base.retrieval_scale_mult + committed.retrieval_scale_mult_delta
    r = max(0.84, min(1.09, r))
    rf = base.relevance_floor_delta + committed.relevance_floor_delta
    rf = max(-0.022, min(0.045, rf))
    return FeedbackLoopAdjustments(r, rf)


def adaptive_prune_integer_delta(workspace_root: Path | None, strategy_id: str) -> int:
    """Small integer prune_k bump from promoted adaptive state (clamped to ±1)."""
    from .adaptive_feedback.constants import adaptive_disabled
    from .adaptive_feedback.state_store import load_committed

    if adaptive_disabled() or workspace_root is None or not strategy_id.strip():
        return 0
    com = load_committed(workspace_root)
    if com is None:
        return 0
    raw = float(com.prune_bias_ema_delta.get(strategy_id.strip(), 0.0))
    step = int(round(raw))
    return max(-1, min(1, step))


def adaptive_skill_prior_nudge(workspace_root: Path | None, skill_id: str | None) -> float:
    """Additive ranking nudge for TOOL-tier skill id from adaptive committed state."""
    from .adaptive_feedback.constants import adaptive_disabled
    from .adaptive_feedback.state_store import load_committed

    if adaptive_disabled() or workspace_root is None:
        return 0.0
    sid = (skill_id or "").strip()
    if not sid:
        return 0.0
    com = load_committed(workspace_root)
    if com is None:
        return 0.0
    return float(com.skill_prior_delta.get(sid, 0.0))
