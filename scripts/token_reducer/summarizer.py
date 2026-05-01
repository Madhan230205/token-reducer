"""Extractive context summary for Claude (local, no APIs)."""

from __future__ import annotations

from pathlib import Path

from .chunker import tokenize
from .intent import IntentType
from .models import Candidate


def summarize_context(
    query: str,
    intent: IntentType,
    candidates: list[Candidate],
) -> tuple[str, str]:
    """Return (short summary, cross-file relationship blurb)."""
    if not candidates:
        return "No indexed context matched this query.", ""

    names = [Path(c.source).name for c in candidates[:5]]
    qtok = set(tokenize(query)[:20])
    overlap_hits = 0
    for c in candidates[:5]:
        tt = set(tokenize(c.text))
        if qtok & tt:
            overlap_hits += 1

    summary = (
        f"Intent={intent}. Retrieved {len(candidates)} chunk(s) from {len({c.source for c in candidates})} file(s). "
        f"Keyword overlap in top results: {overlap_hits}/min(5,{len(candidates)}). "
        f"Primary sources: {', '.join(names)}."
    )

    roots = sorted({str(Path(c.source).parent) for c in candidates[:8]})
    if len(roots) == 1:
        rel = (
            f"All snippets sit under `{roots[0]}`. Trace symbols across these files before editing."
        )
    else:
        rel = f"Spans {len(roots)} directories; likely cross-module flow between: " + "; ".join(
            roots[:4]
        )

    return summary, rel


def why_relevant(candidate: Candidate, query: str, intent: IntentType) -> str:
    parts = [
        f"intent={intent}",
        f"bm25_rank={candidate.fts_rank}",
        f"vec={candidate.vector_score:.3f}",
        f"struct={candidate.structural_score:.3f}",
        f"final={candidate.final_score:.3f}",
    ]
    return "; ".join(parts)
