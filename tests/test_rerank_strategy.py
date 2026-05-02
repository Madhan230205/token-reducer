from __future__ import annotations

from token_reducer.config import configure_rrf
from token_reducer.models import Candidate
from token_reducer.retriever import rerank_candidates


def _c(
    cid: int,
    src: str,
    *,
    bm25: float,
    fts_rank: int,
    vec: float,
) -> Candidate:
    return Candidate(
        chunk_id=cid,
        source=src,
        chunk_index=0,
        text="alpha token here\n",
        token_estimate=10,
        bm25_score=bm25,
        fts_rank=fts_rank,
        vector_score=vec,
        vector_rank=fts_rank,
    )


def test_lexical_heavy_prefers_stronger_lexical_signal() -> None:
    configure_rrf(False)
    try:
        a = _c(1, "a.py", bm25=-0.2, fts_rank=1, vec=0.0)
        b = _c(2, "b.py", bm25=-80.0, fts_rank=2, vec=0.0)
        top, _ = rerank_candidates(
            "alpha token",
            [a, b],
            [],
            top_k=2,
            strategy="lexical_heavy",
        )
        assert top[0].chunk_id == 1
    finally:
        configure_rrf(True)


def test_semantic_heavy_runs_without_error_when_vector_hits_present() -> None:
    """semantic_heavy adjusts weights; exact winner depends on global weight table."""
    configure_rrf(False)
    try:
        fts = _c(1, "a.py", bm25=-50.0, fts_rank=1, vec=0.0)
        vec = _c(2, "b.py", bm25=-60.0, fts_rank=2, vec=0.99)
        top, pool = rerank_candidates(
            "alpha token",
            [fts],
            [vec],
            top_k=2,
            strategy="semantic_heavy",
        )
        assert len(top) == 2
        assert len(pool) == 2
    finally:
        configure_rrf(True)


def test_overlap_heavy_boosts_query_overlap_candidate() -> None:
    configure_rrf(False)
    try:
        fts_hi = _c(1, "a.py", bm25=-1.0, fts_rank=1, vec=0.0)
        fts_hi.text = "foo bar specialtoken zz\n"
        fts_lo = _c(2, "b.py", bm25=-2.0, fts_rank=2, vec=0.0)
        fts_lo.text = "unrelated text\n"
        top, _ = rerank_candidates(
            "specialtoken",
            [fts_hi, fts_lo],
            [],
            top_k=2,
            strategy="overlap_heavy",
        )
        assert top[0].chunk_id == 1
    finally:
        configure_rrf(True)


def test_balanced_hybrid_runs_with_vector_hits() -> None:
    configure_rrf(False)
    try:
        fts = _c(1, "a.py", bm25=-10.0, fts_rank=1, vec=0.0)
        vec = _c(2, "b.py", bm25=-20.0, fts_rank=2, vec=0.95)
        top, pool = rerank_candidates(
            "alpha token",
            [fts],
            [vec],
            top_k=2,
            strategy="balanced_hybrid",
        )
        assert len(top) == 2
        assert len(pool) == 2
    finally:
        configure_rrf(True)
