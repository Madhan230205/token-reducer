"""Outcome-first copy + light transparency (no LLM)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .intent import IntentType
from .models import Candidate


def build_focus_line(
    query: str,
    intent: IntentType,
    selected: list[Candidate],
    *,
    strategy_id: str | None = None,
) -> str:
    """One user-facing line: what was kept and why it matters (not pipeline jargon)."""
    _ = query
    if not selected:
        return "No excerpts matched closely enough; widen paths or rephrase the ask."
    uniq = sorted({Path(c.source).name for c in selected})
    shown = ", ".join(uniq[:5])
    if len(uniq) > 5:
        shown += f", +{len(uniq) - 5} more"
    n = len(selected)
    sid = (strategy_id or "").replace("_", " ").strip()
    mode = f" ({sid})" if sid else ""
    return (
        f"Kept {n} excerpt{'s' if n != 1 else ''} from {len(uniq)} file{'s' if len(uniq) != 1 else ''} "
        f"most aligned with a {intent} workflow{mode}: {shown}."
    )


def build_agent_trace(state: Any) -> list[dict[str, Any]]:
    """Named stages for the agentic path (inspectable, lightweight)."""
    rows: list[dict[str, Any]] = [
        {"stage": "intent", "detail": str(getattr(state, "intent", "")), "status": "ok"},
        {
            "stage": "retrieval",
            "detail": (
                f"lexical_hits={len(getattr(state, 'fts_hits', []) or [])} "
                f"semantic_hits={len(getattr(state, 'vector_hits', []) or [])} "
                f"path={getattr(state, 'vector_retrieval_path', '')}"
            ),
            "status": "ok",
        },
    ]
    cs = getattr(state, "context_strategy", None) or {}
    if isinstance(cs, dict) and cs.get("strategy_id"):
        rows.append(
            {
                "stage": "context_strategy",
                "detail": str(cs.get("strategy_id")),
                "status": "ok",
            }
        )
    sub = getattr(state, "subagent_debug", None) or {}
    ran = sub.get("subagents") if isinstance(sub, dict) else None
    if isinstance(ran, list) and ran:
        rows.append(
            {
                "stage": "context_agents",
                "detail": " → ".join(str(x) for x in ran),
                "status": "ok",
            }
        )
    bullets = getattr(state, "bullets", []) or []
    rows.append({"stage": "compress", "detail": f"bullets={len(bullets)}", "status": "ok"})
    rows.append({"stage": "validate", "detail": "packet_ready", "status": "ok"})
    return rows


def chunk_transparency_rows(selected: list[Candidate]) -> list[dict[str, Any]]:
    """Per-chunk why + score (trust layer for JSON / CLI)."""
    out: list[dict[str, Any]] = []
    for c in selected:
        signals: list[str] = []
        if float(c.fts_score) > 1e-6:
            signals.append("lexical")
        if float(c.vector_score) > 1e-6:
            signals.append("semantic")
        if float(c.overlap_score) > 1e-6:
            signals.append("overlap")
        if float(c.structural_score) > 1e-6:
            signals.append("structure")
        why = "+".join(signals) if signals else "blended_rank"
        out.append(
            {
                "chunk_id": int(c.chunk_id),
                "file": str(c.source),
                "why": why,
                "final_score": round(float(c.final_score), 5),
                "fts_score": round(float(c.fts_score), 5),
                "vector_score": round(float(c.vector_score), 5),
            }
        )
    return out
