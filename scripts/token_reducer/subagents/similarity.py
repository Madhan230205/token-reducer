"""Bounded token-set similarity for subagents (no embeddings, O(n) per pair)."""

from __future__ import annotations

from ..chunker import tokenize


def jaccard_text(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / float(union) if union else 0.0
