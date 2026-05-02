from __future__ import annotations

from token_reducer.intent import detect_intent
from token_reducer.models import Candidate
from token_reducer.subagents.policy import effective_chunk_token_budget, model_scale
from token_reducer.subagents.router import run_subagents


def test_model_scale_haiku_opus() -> None:
    assert model_scale("haiku-3") == 0.7
    assert model_scale("opus-4") == 1.08
    assert model_scale(None) == 1.0


def test_run_subagents_reduces_or_stable() -> None:
    chunks = [
        Candidate(
            chunk_id=i,
            source="a.py",
            chunk_index=i,
            text=f"token alpha beta line {i}\n" * 3,
            token_estimate=40,
            final_score=0.9 - i * 0.01,
        )
        for i in range(8)
    ]
    si = detect_intent("alpha beta context")
    out, meta = run_subagents(chunks, "alpha beta context", si, {"model": "haiku"})
    assert meta["chunks_before"] <= 18
    assert meta["chunks_after"] <= meta["chunks_before"]
    assert set(meta) >= {"subagents", "tokens_before", "tokens_after", "tokens_saved"}
    budget = effective_chunk_token_budget(si["token_budget"], "haiku")
    assert sum(int(c.token_estimate) for c in out) <= budget + 128


def test_effective_chunk_budget_respects_model() -> None:
    assert effective_chunk_token_budget(2000, "haiku") < effective_chunk_token_budget(2000, "opus")
