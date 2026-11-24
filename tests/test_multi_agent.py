"""Multi-agent specialists wrap orchestrator stages (integration smoke)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from token_reducer.context_pipeline import process_prompt
from token_reducer.db import connect_db, upsert_document
from token_reducer.multi_agent import ORDERED_AGENTS, run_multi_agent_compression_phases
from token_reducer.orchestrator import ContextPipelineOrchestrator, ContextRunState
from token_reducer.plugin_settings import get_runtime_defaults


def _seed(conn, tmp: Path) -> None:
    p = tmp / "svc.py"
    p.write_text(
        "def validate_jwt(token: str) -> bool:\n    return bool(token)\n",
        encoding="utf-8",
    )
    t = p.read_text()
    upsert_document(
        conn,
        str(p),
        raw_text=t,
        cleaned_text=t,
        chunk_size_words=24,
        overlap_words=0,
        dimensions=64,
        embedding_backend="hash",
        embedding_model=None,
    )
    conn.commit()


def test_ordered_multi_agent_registry() -> None:
    names = [cls.name for cls in ORDERED_AGENTS]
    assert names == ["cost_optimizer", "retriever", "reasoning_enhancer", "compressor"]


def test_run_multi_agent_phases_populates_trace(tmp_path: Path) -> None:
    dbp = tmp_path / "idx.db"
    conn = connect_db(dbp)
    _seed(conn, tmp_path)
    rt = get_runtime_defaults()
    st = ContextRunState(
        conn=conn,
        db_path=dbp,
        query="jwt validation",
        intent="explain_code",
        runtime=rt,
        memory_blob={},
        session_id="ma1",
        top_k=6,
        fts_k=8,
        vector_k=0,
        min_fts_hits=3,
        hybrid_mode="fallback",
        embedding_backend="hash",
        embedding_model=None,
        dimensions=64,
        word_budget=200,
        relevance_floor=0.12,
        workspace_root=tmp_path,
    )
    orch = ContextPipelineOrchestrator(st)
    with patch("token_reducer.orchestrator.infer_retrieval_tier", return_value="fts_only"):
        rows = run_multi_agent_compression_phases(orch)
    conn.close()
    assert [r["agent"] for r in rows] == ["cost_optimizer", "retriever", "reasoning_enhancer", "compressor"]
    assert rows[1].get("fts_hits", 0) >= 1
    assert rows[3].get("has_claude_context") is True


def test_process_prompt_debug_includes_multi_agent(tmp_path: Path) -> None:
    dbp = tmp_path / "idx.db"
    conn = connect_db(dbp)
    _seed(conn, tmp_path)
    rt = get_runtime_defaults()
    st = ContextRunState(
        conn=conn,
        db_path=dbp,
        query="jwt",
        intent="explain_code",
        runtime=rt,
        memory_blob={},
        session_id="ma2",
        top_k=6,
        fts_k=8,
        vector_k=0,
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
        out = process_prompt("jwt validation flow", {"context_run_state": st}, debug=True)
    conn.close()
    ma = out["debug"].get("multi_agent")
    assert isinstance(ma, list)
    assert len(ma) == 4
