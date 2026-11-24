from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from token_reducer.db import connect_db, upsert_document
from token_reducer.models import Candidate
from token_reducer.ranking import apply_claude_intent_rerank


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    conn = connect_db(tmp_path / "r.db")
    upsert_document(
        conn=conn,
        source=str(tmp_path / "a.py"),
        raw_text="def foo():\n    raise ValueError('x')\n",
        cleaned_text="def foo():\n    raise ValueError('x')\n",
        chunk_size_words=80,
        overlap_words=0,
        dimensions=64,
        embedding_backend="hash",
        embedding_model=None,
    )
    conn.commit()
    yield conn
    conn.close()


def test_intent_rerank_orders_bug_fix_toward_errors(db: sqlite3.Connection) -> None:
    c_err = Candidate(
        chunk_id=1,
        source="a.py",
        chunk_index=0,
        text="def foo():\n    raise ValueError('bad')\n",
        token_estimate=10,
        bm25_score=-1.0,
        fts_rank=2,
        vector_score=0.1,
        fts_score=0.2,
    )
    c_plain = Candidate(
        chunk_id=2,
        source="a.py",
        chunk_index=1,
        text="x = 1\ny = 2\n",
        token_estimate=5,
        bm25_score=-0.5,
        fts_rank=1,
        vector_score=0.5,
        fts_score=0.6,
    )
    out = apply_claude_intent_rerank(db, [c_plain, c_err], "fix error in foo", "bug_fix")
    by_id = {c.chunk_id: c for c in out}
    assert by_id[c_err.chunk_id].structural_score > by_id[c_plain.chunk_id].structural_score


def test_rrf_scores_are_normalised_to_unit_top() -> None:
    from token_reducer.retriever import reciprocal_rank_fusion

    def cand(cid: int) -> Candidate:
        return Candidate(
            chunk_id=cid, source=f"{cid}.py", chunk_index=0, text="t", token_estimate=1
        )

    ranked = reciprocal_rank_fusion([cand(1), cand(2), cand(3)], [cand(2), cand(4)])
    assert ranked[0].chunk_id == 2  # appears in both lists
    assert ranked[0].final_score == pytest.approx(1.0)
    assert all(0.0 < c.final_score <= 1.0 for c in ranked)
    # a 0.15-0.18 relevance floor must not wipe out the single-list candidates
    assert sum(c.final_score >= 0.18 for c in ranked) == len(ranked)
