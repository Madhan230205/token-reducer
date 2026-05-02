from __future__ import annotations

from token_reducer.audit_spine import (
    apply_retrieval_and_routing,
    bootstrap_spine,
    compute_retrieval_disagreement,
    new_audit_spine,
    scored_pool_to_candidates,
)
from token_reducer.execution_route import ExecutionRoute
from token_reducer.intent import detect_intent
from token_reducer.models import Candidate


def _c(
    chunk_id: int,
    *,
    fts_rank: int | None = None,
    vector_rank: int | None = None,
    fts_score: float = 0.0,
    vector_score: float = 0.0,
    final_score: float = 0.0,
) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        source=f"file{chunk_id}.py",
        chunk_index=0,
        text="x",
        token_estimate=10,
        bm25_score=fts_score,
        fts_rank=fts_rank,
        vector_rank=vector_rank,
        fts_score=fts_score,
        vector_score=vector_score,
        final_score=final_score,
    )


def test_new_audit_spine_has_schema_and_trace_id() -> None:
    spine, tid, ts = new_audit_spine()
    assert spine.schema_version == "audit_spine_v0_1"
    assert spine.trace_id == tid
    assert spine.timestamp == ts
    assert len(tid) == 36


def test_compute_disagreement_no_vector_is_neutral() -> None:
    fts = [_c(1, fts_rank=1, fts_score=0.9), _c(2, fts_rank=2, fts_score=0.8)]
    d = compute_retrieval_disagreement(fts, [])
    assert d.fts_vector_overlap == 1.0
    assert d.rank_correlation == 1.0
    assert not d.low_overlap_flag
    assert not d.mismatch_flag


def test_compute_disagreement_overlap_and_flags() -> None:
    fts = [_c(i, fts_rank=i, fts_score=1.0 / i) for i in range(1, 11)]
    vec = [_c(i + 100, vector_rank=i, vector_score=1.0 / i) for i in range(1, 11)]
    d = compute_retrieval_disagreement(fts, vec, top_n=10)
    assert d.fts_vector_overlap == 0.0
    assert d.low_overlap_flag
    assert d.mismatch_flag


def test_scored_pool_candidate_sources() -> None:
    scored = [
        _c(1, fts_rank=1, vector_rank=1, fts_score=0.2, vector_score=0.3, final_score=0.5),
        _c(2, fts_rank=2, fts_score=0.4, final_score=0.4),
    ]
    rows = scored_pool_to_candidates(scored, top_n=10)
    assert rows[0].source == "hybrid"
    assert rows[1].source == "fts"


def test_bootstrap_and_retrieval_routing_roundtrip() -> None:
    si = detect_intent("fix the bug in auth")
    route = ExecutionRoute(
        tier="complex",
        skill_id=None,
        skip_subagents=False,
        skip_lsp=False,
        retrieval_scale=1.0,
        reducer_token_threshold=2800,
        efficiency_score=None,
    )
    spine = bootstrap_spine("fix the bug in auth", si, route)
    fts = [_c(1, fts_rank=1, fts_score=0.9)]
    vec = [_c(2, vector_rank=1, vector_score=0.9)]
    scored = [_c(1, fts_rank=1, vector_rank=None, fts_score=0.9, final_score=0.95)]
    spine2 = apply_retrieval_and_routing(
        spine,
        scored_pool=scored,
        fts_hits=fts,
        vector_hits=vec,
        route=route,
        context_decision=None,
    )
    assert spine2.routing.selected_tier == "complex"
    assert spine2.retrieval.candidates
    assert spine2.retrieval.disagreement.fts_vector_overlap == 0.0
