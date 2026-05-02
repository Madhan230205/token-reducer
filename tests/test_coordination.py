from __future__ import annotations

from token_reducer.models import Candidate
from token_reducer.subagents.coordination import (
    build_run_memory,
    decompose_tasks,
    diversify_chunk_order,
    extract_focus_paths,
    extract_focus_terms,
    merge_subtask_streams,
)


def test_extract_focus_paths_multi_segment() -> None:
    q = "crash happens in src/services/auth/handler.py when validating JWT tokens"
    paths = extract_focus_paths(q)
    assert any("handler.py" in p for p in paths)


def test_extract_focus_terms_drops_stopwords() -> None:
    terms = extract_focus_terms("how does the frobnicateWidget handle retries for users")
    assert "frobnicateWidget" in terms or "retries" in terms
    assert "the" not in terms


def test_decompose_bug_fix_has_multiple_steps() -> None:
    steps = decompose_tasks(
        "stack trace on deploy",
        {"legacy_intent": "bug_fix", "type": "code", "k": 40, "token_budget": 2000, "compression_level": "high"},
    )
    assert len(steps) >= 2
    assert any("error" in s.lower() or "symptom" in s.lower() for s in steps)


def test_merge_subtask_streams_interleaves_path_and_terms() -> None:
    rm = {
        "focus_paths": ["src/services/auth.py"],
        "focus_terms": ["jwt"],
    }
    chunks = [
        Candidate(
            chunk_id=1,
            source="src/other/util.py",
            chunk_index=0,
            text="helper",
            token_estimate=5,
            final_score=0.95,
        ),
        Candidate(
            chunk_id=2,
            source="src/services/auth.py",
            chunk_index=0,
            text="jwt validate",
            token_estimate=5,
            final_score=0.9,
        ),
        Candidate(
            chunk_id=3,
            source="src/api/routes.py",
            chunk_index=0,
            text="jwt middleware",
            token_estimate=5,
            final_score=0.85,
        ),
    ]
    merged = merge_subtask_streams(chunks, rm)
    assert {c.chunk_id for c in merged} == {1, 2, 3}
    assert merged[0].chunk_id in (2, 3)


def test_diversify_chunk_order_is_stable() -> None:
    def c(i: int, src: str, score: float) -> Candidate:
        return Candidate(
            chunk_id=i,
            source=src,
            chunk_index=0,
            text="x",
            token_estimate=1,
            final_score=score,
        )

    base = [c(i, f"f{i}.py", 1.0 - i * 0.01) for i in range(8)]
    out1 = diversify_chunk_order(list(base), "same prompt seed")
    out2 = diversify_chunk_order(list(base), "same prompt seed")
    assert [x.chunk_id for x in out1] == [x.chunk_id for x in out2]


def test_build_run_memory_keys() -> None:
    rm = build_run_memory(
        "see src/api/x.py for the WidgetFactory pattern",
        {"legacy_intent": "explain_code", "type": "analysis", "k": 40, "token_budget": 2000, "compression_level": "medium"},
    )
    assert "decomposition" in rm and "focus_paths" in rm and "focus_terms" in rm
    assert isinstance(rm["decomposition"], list)
