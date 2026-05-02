"""Light integration: policy + rerank differ by task under controlled tier."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from token_reducer.context_pipeline import process_prompt
from token_reducer.db import connect_db, upsert_document
from token_reducer.intent import IntentType
from token_reducer.orchestrator import ContextRunState
from token_reducer.plugin_settings import get_runtime_defaults


def _seed_minimal_index(conn, tmp: Path) -> None:
    p = tmp / "app.py"
    p.write_text("def main():\n    print('hello')\n", encoding="utf-8")
    t = p.read_text()
    upsert_document(
        conn,
        str(p),
        raw_text=t,
        cleaned_text=t,
        chunk_size_words=20,
        overlap_words=0,
        dimensions=64,
        embedding_backend="hash",
        embedding_model=None,
    )
    conn.commit()


def test_pipeline_rerank_strategy_navigation_vs_refactor_under_fts_only(tmp_path: Path) -> None:
    dbp = tmp_path / "idx.db"
    conn = connect_db(dbp)
    _seed_minimal_index(conn, tmp_path)
    rt = get_runtime_defaults()

    def run(intent: IntentType, query: str) -> str:
        st = ContextRunState(
            conn=conn,
            db_path=dbp,
            query=query,
            intent=intent,
            runtime=rt,
            memory_blob={},
            session_id="s1",
            top_k=6,
            fts_k=8,
            vector_k=4,
            min_fts_hits=3,
            hybrid_mode="fallback",
            embedding_backend="hash",
            embedding_model=None,
            dimensions=64,
            word_budget=200,
            relevance_floor=0.12,
            workspace_root=tmp_path,
        )
        with patch("token_reducer.orchestrator.infer_retrieval_tier", return_value="fts_only"):
            process_prompt(query, {"context_run_state": st})
        assert st.policy is not None
        return st.policy.rerank_strategy

    nav = run("navigation", "where is the entry point")
    ref = run("refactor", "extract helper from main")
    assert nav == "lexical_heavy"
    assert ref == "overlap_heavy"
    conn.close()
