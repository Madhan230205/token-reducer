"""Post-ranking score adjustments from retrieval plan (narrow boosts, not re-embedding)."""

from __future__ import annotations

from .models import Candidate


def apply_source_boost(
    pool: list[Candidate],
    boosted: frozenset[str],
    *,
    delta: float = 0.035,
    patch_first: bool = False,
) -> list[Candidate]:
    """Slightly raise final_score for planned high-value paths, then re-sort."""
    if not boosted or not pool:
        return pool
    eff = delta + (0.02 if patch_first else 0.0)
    for c in pool:
        if c.source in boosted:
            c.final_score = min(1.0, float(c.final_score) + eff)
    pool.sort(key=lambda x: x.final_score, reverse=True)
    return pool
