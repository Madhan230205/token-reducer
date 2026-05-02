"""Tests for adaptive collector and feedback merge."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from token_reducer.adaptive_feedback.collector import build_events_from_run, cohort_key_from_state
from token_reducer.adaptive_feedback.models import CommittedActuators, SignalType
from token_reducer.adaptive_feedback.state_store import promote_committed_atomic
from token_reducer.execution_route import ExecutionRoute
from token_reducer.feedback import FeedbackLoopAdjustments, feedback_loop_adjustments_with_adaptive
from token_reducer.models import Candidate
from token_reducer.orchestrator import ContextRunState
from token_reducer.plugin_settings import get_runtime_defaults


def _minimal_state(tmp_path: Path, **kwargs: object) -> ContextRunState:
    conn = sqlite3.connect(":memory:")
    rt = get_runtime_defaults()
    defaults: dict[str, object] = {
        "conn": conn,
        "db_path": tmp_path / "db.sqlite",
        "query": "hello",
        "intent": "explain_code",
        "runtime": rt,
        "memory_blob": {},
        "session_id": "s1",
        "top_k": 12,
        "fts_k": 20,
        "vector_k": 8,
        "min_fts_hits": 2,
        "hybrid_mode": "auto",
        "embedding_backend": "disabled",
        "embedding_model": None,
        "dimensions": 384,
        "word_budget": 8000,
        "relevance_floor": 0.12,
        "workspace_root": tmp_path,
        "execution_route": ExecutionRoute(
            tier="complex",
            skill_id=None,
            skip_subagents=False,
            skip_lsp=False,
            retrieval_scale=1.0,
            reducer_token_threshold=2800,
            efficiency_score=None,
        ),
        "scored_pool": [],
        "retrieval_retry_done": False,
        "context_strategy": {"strategy_id": "balanced"},
    }
    defaults.update(kwargs)
    return ContextRunState(**defaults)  # type: ignore[arg-type]


def test_cohort_key_from_state(tmp_path: Path) -> None:
    st = _minimal_state(tmp_path)
    assert cohort_key_from_state(st) == ("complex", "balanced", "", "explain_code")


def test_build_events_weak_pool(tmp_path: Path) -> None:
    c = Candidate(
        chunk_id=1,
        source="a.py",
        chunk_index=0,
        text="x",
        token_estimate=3,
        final_score=0.05,
    )
    st = _minimal_state(tmp_path, scored_pool=[c])
    evs = build_events_from_run(st)
    assert len(evs) == 1
    assert evs[0].signal_type == SignalType.RETRIEVAL_MISS_WEAK_POOL


def test_build_events_hit_strong(tmp_path: Path) -> None:
    c = Candidate(
        chunk_id=1,
        source="a.py",
        chunk_index=0,
        text="x",
        token_estimate=3,
        final_score=0.4,
    )
    st = _minimal_state(tmp_path, scored_pool=[c, c])
    evs = build_events_from_run(st)
    assert evs[0].signal_type == SignalType.RETRIEVAL_HIT_STRONG


def test_feedback_merge_layer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TOKEN_REDUCER_ADAPT_DISABLE", raising=False)
    promote_committed_atomic(
        tmp_path,
        CommittedActuators(retrieval_scale_mult_delta=0.03, relevance_floor_delta=0.01),
    )
    merged = feedback_loop_adjustments_with_adaptive(
        log_path=tmp_path / "missing.jsonl",
        workspace_root=tmp_path,
    )
    assert isinstance(merged, FeedbackLoopAdjustments)
    assert merged.retrieval_scale_mult == pytest.approx(1.03)
    assert merged.relevance_floor_delta == pytest.approx(0.01)
