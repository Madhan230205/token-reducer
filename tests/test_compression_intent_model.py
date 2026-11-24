from __future__ import annotations

from token_reducer.compressor import importance_rank_indices
from token_reducer.model_tuning import compression_model_tuning
from token_reducer.models import Candidate


def _c(chunk_id: int, text: str, score: float) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        source="x.py",
        chunk_index=chunk_id,
        text=text,
        token_estimate=40,
        final_score=score,
    )


def test_bug_fix_prioritizes_exception_like_chunk() -> None:
    q = "crash when saving user"
    a = _c(0, "def save(): pass", 0.55)
    b = _c(1, "raise ValueError('missing user')", 0.55)
    order = importance_rank_indices([a, b], q, legacy_intent="bug_fix")
    assert order[0] == 1


def test_haiku_compression_tuning_stricter_floor_than_opus() -> None:
    h = compression_model_tuning("haiku")
    o = compression_model_tuning("opus-4")
    assert h.budget_factor < o.budget_factor
    assert h.relevance_floor_shift > o.relevance_floor_shift
