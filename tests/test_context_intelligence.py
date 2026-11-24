"""Unit tests for context intelligence heuristics and ranking precedence."""

from __future__ import annotations

from token_reducer.context_intelligence.heuristic import build_decision_d1, intel_retry_allowed
from token_reducer.context_intelligence.models import ContextDecision
from token_reducer.context_intelligence.ranking import validate_ranking_weights
from token_reducer.context_intelligence.strategy_nudge import apply_task_shape_nudge
from token_reducer.context_pipeline import rerank_chunks
from token_reducer.intent import detect_intent, structured_intent_to_dict


def _candidate(cid: int, final_score: float, text: str = "alpha beta gamma"):
    from token_reducer.models import Candidate

    return Candidate(
        chunk_id=cid,
        source=f"x{cid}.py",
        chunk_index=0,
        text=text,
        token_estimate=8,
        final_score=final_score,
        fts_score=final_score,
        vector_score=0.0,
    )


def test_validate_ranking_weights_accepts_normalized_triple() -> None:
    w = validate_ranking_weights({"semantic": 0.5, "keyword": 0.25, "recency": 0.25})
    assert w == {"semantic": 0.5, "keyword": 0.25, "recency": 0.25}


def test_validate_ranking_weights_rejects_bad_sum() -> None:
    assert validate_ranking_weights({"semantic": 0.9, "keyword": 0.9, "recency": 0.9}) is None


def test_apply_task_shape_nudge_respects_bounds() -> None:
    m, p = apply_task_shape_nudge(50, 15, "generation")
    assert 40 <= m <= 60
    assert 10 <= p <= 18


def test_build_decision_d1_id_stable_for_fixed_signals() -> None:
    d0 = ContextDecision(task_family="explain_code", task_shape="analysis", ranking_profile="default")
    scores = {
        "bug_fix": 0,
        "feature_add": 0,
        "explain_code": 3,
        "refactor": 0,
        "navigation": 0,
    }
    pool = [_candidate(1, 0.9), _candidate(2, 0.85), _candidate(3, 0.8)]
    d1a = build_decision_d1(
        d0,
        query="how does routing work",
        scores=scores,
        fts_hits=10,
        vector_hits=0,
        min_fts_hits=3,
        scored_pool=pool,
        weak_pool_legacy=False,
    )
    d1b = build_decision_d1(
        d0,
        query="how does routing work",
        scores=scores,
        fts_hits=10,
        vector_hits=0,
        min_fts_hits=3,
        scored_pool=pool,
        weak_pool_legacy=False,
    )
    assert d1a.decision_id == d1b.decision_id


def test_intel_retry_allowed_only_high_risk_with_evidence() -> None:
    hi_weak = ContextDecision(
        risk_level="high",
        signals={"weak_retrieval": True, "top3_spread": 0.01},
    )
    assert intel_retry_allowed(hi_weak, weak_pool_legacy=False) is True

    med = ContextDecision(
        risk_level="medium",
        signals={"weak_retrieval": True, "top3_spread": 0.01},
    )
    assert intel_retry_allowed(med, weak_pool_legacy=True) is False

    hi_strong_spread = ContextDecision(
        risk_level="high",
        signals={"weak_retrieval": False, "top3_spread": 0.5},
    )
    assert intel_retry_allowed(hi_strong_spread, weak_pool_legacy=False) is False


def test_rerank_chunks_profile_changes_blend() -> None:
    intent = structured_intent_to_dict(detect_intent("explore alpha beta"))
    chunks = [
        _candidate(1, 0.35, "alpha beta raretoken"),
        _candidate(2, 0.92, "unrelated noise token blob"),
        _candidate(3, 0.55, "alpha beta gamma delta"),
    ]
    out_lex = rerank_chunks(list(chunks), "alpha beta gamma", intent, conn=None, ranking_profile="lexical_heavy")
    out_sem = rerank_chunks(list(chunks), "alpha beta gamma", intent, conn=None, ranking_profile="semantic_heavy")
    assert any(
        abs(a.final_score - b.final_score) > 1e-6
        for a, b in zip(out_lex, out_sem, strict=True)
    )


def test_rerank_chunks_invalid_weights_use_profile() -> None:
    intent = structured_intent_to_dict(detect_intent("x"))
    chunks = [_candidate(1, 0.5), _candidate(2, 0.4)]
    a = rerank_chunks(list(chunks), "alpha beta", intent, conn=None, ranking_profile="default")
    b = rerank_chunks(
        list(chunks),
        "alpha beta",
        intent,
        conn=None,
        ranking_profile="default",
        ranking_weights={"semantic": 0.99, "keyword": 0.99},
    )
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x.chunk_id == y.chunk_id
        assert abs(x.final_score - y.final_score) < 1e-9
