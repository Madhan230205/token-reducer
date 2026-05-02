"""Hybrid explainability — deterministic one-line model nudge + compact sidecar.

Spec: docs/superpowers/specs/2026-05-02-plugin-explainability-hybrid-ux-design.md
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .audit_spine import AuditSpine

SCHEMA_VERSION = "explain_hybrid_v1"
MAX_ONE_LINER_CHARS = 180


def confidence_band(spine: dict[str, Any]) -> str:
    out = spine.get("outcome") or {}
    raw_fc = out.get("final_confidence")
    fc = float(raw_fc if raw_fc is not None else 0.62)
    disc = (spine.get("retrieval") or {}).get("disagreement") or {}
    if disc.get("mismatch_flag"):
        fc -= 0.08
    if disc.get("low_overlap_flag"):
        fc -= 0.05
    fc = max(0.0, min(1.0, fc))
    if fc >= 0.72:
        return "high"
    if fc >= 0.48:
        return "medium"
    return "low"


def preset_label(subagent_profile_used: str | None) -> str:
    p = (subagent_profile_used or "full").strip().lower()
    if p in ("lean", "none"):
        return "fast"
    return "balanced"


def _chunk_id_to_path(spine: dict[str, Any]) -> dict[str, str]:
    m: dict[str, str] = {}
    for c in (spine.get("retrieval") or {}).get("candidates") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("chunk_id", ""))
        path = str(c.get("path") or "")
        if cid:
            m[cid] = path
    return m


def focus_phrase(spine: dict[str, Any]) -> str:
    id_to_path = _chunk_id_to_path(spine)
    names: list[str] = []
    for cid in (spine.get("compression") or {}).get("retained_chunks") or []:
        raw = id_to_path.get(str(cid), "")
        if raw:
            names.append(Path(raw).name)
        if len(names) >= 4:
            break
    dedup: list[str] = []
    for n in names:
        if n not in dedup:
            dedup.append(n)
    if not dedup:
        tier = (spine.get("routing") or {}).get("selected_tier") or "general"
        return f"{tier} context"
    return " + ".join(dedup[:3])


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def render_payload_one_liner(
    spine: dict[str, Any],
    *,
    preset: str,
) -> str:
    band = confidence_band(spine)
    focus = _truncate_words(focus_phrase(spine), 12)
    line = f"Context optimized ({preset}, {band} confidence). Key focus: {focus}."
    if len(line) > MAX_ONE_LINER_CHARS:
        line = line[: MAX_ONE_LINER_CHARS - 1].rstrip() + "."
    return line


def render_sidecar_text(
    spine: dict[str, Any],
    *,
    preset: str,
    patch_first: bool,
    focus_paths: tuple[str, ...],
) -> str:
    id_to_path = _chunk_id_to_path(spine)
    fp_set = set(focus_paths)
    retained = list((spine.get("compression") or {}).get("retained_chunks") or [])
    dropped = list((spine.get("compression") or {}).get("dropped_chunks") or [])
    band = confidence_band(spine)

    sel_parts: list[str] = []
    for cid in retained[:3]:
        path = id_to_path.get(str(cid), "")
        label = Path(path).name if path else str(cid)
        reason = "patch path" if patch_first and path and path in fp_set else "top relevance"
        sel_parts.append(f"{label} ({reason})")
    sel_line = "Selected: " + (", ".join(sel_parts) if sel_parts else "(none)")

    drop_parts: list[str] = []
    for cid in dropped[:3]:
        path = id_to_path.get(str(cid), "")
        label = Path(path).name if path else str(cid)
        drop_parts.append(f"{label} (redundant)")
    lines = [sel_line]
    if drop_parts:
        lines.append("Dropped: " + ", ".join(drop_parts))
    lines.append(f"Compression: {preset} • {band} confidence")
    text = "\n".join(lines)
    if len(text.splitlines()) > 6:
        lines = [sel_line, f"Compression: {preset} • {band} confidence"]
        text = "\n".join(lines)
    return text


def render_verbose_bundle(spine: dict[str, Any]) -> dict[str, Any]:
    rc = spine.get("retrieval") or {}
    comp = spine.get("compression") or {}
    return {
        "audit_spine": spine,
        "counts": {
            "retrieval_candidates": len(rc.get("candidates") or []),
            "retained_chunks": len(comp.get("retained_chunks") or []),
            "dropped_chunks": len(comp.get("dropped_chunks") or []),
        },
    }


def attach_hybrid_explainability(
    payload: dict[str, object],
    spine: AuditSpine | None,
    *,
    subagent_profile_used: str | None,
    patch_first: bool,
) -> None:
    """Inject ``context_optimization_line`` + ``context_explainability`` into plugin payload."""
    if spine is None:
        return
    spine_dict = spine.to_jsonable()
    preset = preset_label(subagent_profile_used)
    raw_focus = payload.get("focus_paths")
    focus_paths: tuple[str, ...] = ()
    if isinstance(raw_focus, list):
        focus_paths = tuple(str(x) for x in raw_focus)

    payload["context_optimization_line"] = render_payload_one_liner(spine_dict, preset=preset)
    verbose = None
    flag = os.environ.get("TOKEN_REDUCER_EXPLAIN_VERBOSE", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        verbose = render_verbose_bundle(spine_dict)

    payload["context_explainability"] = {
        "schema_version": SCHEMA_VERSION,
        "sidecar": render_sidecar_text(
            spine_dict,
            preset=preset,
            patch_first=patch_first,
            focus_paths=focus_paths,
        ),
        "verbose": verbose,
    }


def attach_hybrid_explainability_to_context(state: Any) -> None:
    """After outcome is on the spine, enrich ``state.claude_context`` dict."""
    payload = getattr(state, "claude_context", None)
    spine = getattr(state, "audit_spine", None)
    if not isinstance(payload, dict) or spine is None:
        return
    policy = getattr(state, "policy", None)
    patch_first = bool(policy.patch_first) if policy is not None else False
    attach_hybrid_explainability(
        payload,
        spine,
        subagent_profile_used=getattr(state, "subagent_profile_used", None),
        patch_first=patch_first,
    )
