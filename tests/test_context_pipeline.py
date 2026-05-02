from __future__ import annotations

from pathlib import Path

from token_reducer.context_pipeline import rerank_chunks
from token_reducer.db import connect_db
from token_reducer.intent import detect_intent, structured_intent_to_dict
from token_reducer.models import Candidate


def test_detect_intent_has_required_keys() -> None:
    s = detect_intent("Fix crash in handler")
    d = structured_intent_to_dict(s)
    assert d["type"] in ("code", "chat", "analysis")
    assert isinstance(d["k"], int) and d["k"] > 0
    assert isinstance(d["token_budget"], int) and d["token_budget"] > 0
    assert d["compression_level"] in ("high", "medium", "low")
    assert d["legacy_intent"] == "bug_fix"


def test_rerank_chunks_orders_keyword_match_higher() -> None:
    conn = connect_db(Path(":memory:"))
    try:
        a = Candidate(
            chunk_id=1,
            source="x.py",
            chunk_index=0,
            text="unrelated fluff",
            token_estimate=5,
            final_score=0.5,
            vector_score=0.5,
            fts_score=0.5,
        )
        b = Candidate(
            chunk_id=2,
            source="y.py",
            chunk_index=0,
            text="specialtoken alpha beta",
            token_estimate=6,
            final_score=0.5,
            vector_score=0.5,
            fts_score=0.5,
        )
        intent = detect_intent("find specialtoken")
        out = rerank_chunks([a, b], "find specialtoken", intent, conn=conn)
        assert out[0].chunk_id == 2
    finally:
        conn.close()
