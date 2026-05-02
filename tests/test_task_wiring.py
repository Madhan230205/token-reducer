from __future__ import annotations

from pathlib import Path

from token_reducer.compressor import (
    compress_candidates,
    extract_error_log_lines,
    extract_invariant_snippets,
    lines_touching_tokens,
)
from token_reducer.models import Candidate
from token_reducer.verify_loop import build_verification_plan, run_shadow_lint_on_files


def test_extract_invariant_snippets_keeps_raise() -> None:
    text = "def f():\n    if True:\n        raise RuntimeError('x')\n"
    snips = extract_invariant_snippets(text)
    assert any("raise" in s for s in snips)


def test_lines_touching_tokens() -> None:
    text = "class Foo:\n    def bar(self):\n        return 1\n"
    lines = lines_touching_tokens(text, frozenset({"Foo", "bar"}))
    assert len(lines) >= 1


def test_build_verification_plan_patch_first() -> None:
    plan = build_verification_plan("refactor")
    assert plan["preferred_output"] == "patch"
    assert plan["max_retries"] >= 1
    assert "failure_recovery" in plan


def test_build_verification_plan_debug_allows_two_retries() -> None:
    plan = build_verification_plan("debug")
    assert plan["max_retries"] == 2
    assert "narrow_retry_hint" in plan
    assert plan["failure_recovery"]["narrow_context_only"] is True


def test_build_verification_plan_echoes_query_for_retry_context() -> None:
    plan = build_verification_plan("refactor", query="rename FooService")
    assert "query_echo_for_retry" in plan


def test_compress_strict_keeps_more_invariants_than_off() -> None:
    text = "def foo():\n    raise ValueError('x')\nassert foo() is None\n"
    c = Candidate(
        chunk_id=1,
        source="t.py",
        chunk_index=0,
        text=text,
        token_estimate=20,
        final_score=0.9,
    )
    off = compress_candidates("error", [c], 200, 0.01, code_invariant_level="off")
    strict = compress_candidates("error", [c], 400, 0.01, code_invariant_level="strict")
    assert sum(len(x) for x in strict) >= sum(len(x) for x in off)


def test_extract_error_log_lines() -> None:
    text = "def f():\n    logger.error('bad')\n    return\n"
    assert any("logger" in x for x in extract_error_log_lines(text))


def test_run_shadow_lint_on_files_noop_without_cmds(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    p.write_text("x = 1\n")
    ok, msg = run_shadow_lint_on_files([p], cmds={}, timeout=5)
    assert ok
