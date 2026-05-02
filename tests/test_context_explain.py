from __future__ import annotations

from token_reducer.context_explain import (
    build_agent_trace,
    build_focus_line,
    chunk_transparency_rows,
)
from token_reducer.models import Candidate


def _c(cid: int, src: str, fts: float, vec: float, fs: float) -> Candidate:
    return Candidate(
        chunk_id=cid,
        source=src,
        chunk_index=0,
        text="body",
        token_estimate=10,
        fts_score=fts,
        vector_score=vec,
        overlap_score=0.0,
        structural_score=0.0,
        final_score=fs,
    )


def test_build_focus_line_lists_files() -> None:
    s = [_c(1, "a/auth.py", 1, 0, 0.9), _c(2, "b/routes.py", 0.5, 0.2, 0.7)]
    line = build_focus_line("fix jwt", "bug_fix", s, strategy_id="failure_adjacent")
    assert "auth.py" in line and "routes.py" in line and "bug_fix" in line


def test_chunk_transparency_lists_signals() -> None:
    rows = chunk_transparency_rows([_c(1, "x.py", 0.2, 0.0, 0.5)])
    assert rows[0]["why"] == "lexical"
    assert rows[0]["chunk_id"] == 1


def test_build_agent_trace_minimal() -> None:
    class S:
        intent = "explain_code"
        fts_hits = []
        vector_hits = []
        vector_retrieval_path = "disabled"
        context_strategy = None
        subagent_debug = None
        bullets = ["a"]

    rows = build_agent_trace(S())
    assert rows[0]["stage"] == "intent"
    assert any(r["stage"] == "compress" for r in rows)
