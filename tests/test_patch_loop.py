from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from token_reducer.models import Candidate
from token_reducer.patch_loop import (
    ApplyOutcome,
    PatchResult,
    apply_patch_result,
    generate_patch,
    run_closed_edit_loop,
)
from token_reducer.verify_loop import VerificationOutcome, run_verification


def test_generate_patch_simple_replace(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    f.write_text("BAD = 1\n", encoding="utf-8")
    pol = MagicMock()
    pol.patch_first = True
    c = Candidate(chunk_id=1, source=str(f), chunk_index=0, text=f.read_text(), token_estimate=10, final_score=0.9)
    p = generate_patch(
        "replace BAD with GOOD",
        pol,
        None,
        [c],
        None,
        workspace_root=tmp_path,
    )
    assert p.applicable
    assert "BAD" in p.patch_text and "GOOD" in p.patch_text
    assert "<<<< SEARCH" in p.patch_text


def test_apply_patch_result_dry_run(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("x = 1\n", encoding="utf-8")
    rel = f.name
    patch = PatchResult(
        target_files=(str(f.resolve()),),
        diff_hunks="",
        rationale="t",
        patch_text=f"<<<< SEARCH {rel}\nx = 1\n==== REPLACE\nx = 2\n>>>>",
        applicable=True,
    )
    out = apply_patch_result(patch, tmp_path, dry_run=True)
    assert out.success
    assert "x = 1" in f.read_text()


def test_apply_patch_result_writes_when_not_dry(tmp_path: Path) -> None:
    f = tmp_path / "b.py"
    f.write_text("x = 1\n", encoding="utf-8")
    rel = f.name
    patch = PatchResult(
        target_files=(str(f.resolve()),),
        diff_hunks="",
        rationale="t",
        patch_text=f"<<<< SEARCH {rel}\nx = 1\n==== REPLACE\nx = 99\n>>>>",
        applicable=True,
    )
    out = apply_patch_result(patch, tmp_path, dry_run=False)
    assert out.success
    assert "99" in f.read_text()


def test_run_verification_catches_syntax_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("def oops(\n", encoding="utf-8")
    v = run_verification([bad], {}, cmds={}, timeout=5)
    assert not v.success
    assert v.errors


def test_closed_loop_skips_without_patch_first(tmp_path: Path) -> None:
    st = MagicMock()
    st.query = "q"
    st.policy = MagicMock(patch_first=False)
    st.retrieval_plan = None
    st.conn = None
    st.runtime = MagicMock(shadow_linter_cmds={})
    r = run_closed_edit_loop(st, workspace_root=tmp_path, dry_run=True)
    assert r["closed_loop"] == "skipped"


def test_max_retries_respected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from token_reducer import patch_loop as pl

    f = tmp_path / "z.py"
    f.write_text("x = 0\n", encoding="utf-8")
    rel = f.name

    def static_patch(*a: object, **kw: object) -> PatchResult:
        return PatchResult(
            target_files=(str(f.resolve()),),
            diff_hunks="",
            rationale="x",
            patch_text=f"<<<< SEARCH {rel}\nx = 0\n==== REPLACE\nx = 1\n>>>>",
            applicable=True,
        )

    monkeypatch.setattr(pl, "generate_patch", static_patch)
    monkeypatch.setattr(
        pl,
        "apply_patch_result",
        lambda *a, **k: ApplyOutcome(True, ("applied",), (str(f.resolve()),)),
    )
    monkeypatch.setattr(
        pl,
        "run_verification",
        lambda *a, **k: VerificationOutcome(False, ("always bad",), (str(f.resolve()),)),
    )

    st = MagicMock()
    st.query = "replace x with y"
    st.policy = MagicMock(
        patch_first=True,
        rewrite_query=True,
        verification_plan={"max_retries": 1, "lint_timeout_seconds": 5},
        boosted_sources=frozenset(),
        must_keep_symbol_tokens=frozenset(),
        effective_word_budget=200,
        effective_relevance_floor=0.1,
        code_invariant_level="standard",
        include_callers=True,
        include_callees=True,
        task_mode="debug",
        rerank_strategy="lexical_heavy",
        retry_on_low_score=True,
    )
    st.retrieval_plan = None
    st.repo_map = None
    st.selected = []
    st.bullets = []
    st.conn = MagicMock()

    def fts(*a: object, **kw: object):
        return [
            Candidate(chunk_id=2, source=str(f), chunk_index=0, text="x=0", token_estimate=5, final_score=0.5)
        ]

    monkeypatch.setattr(pl, "fts_retrieve", fts)

    r = run_closed_edit_loop(st, workspace_root=tmp_path, dry_run=True)
    assert r["verification"]["status"] == "failed"
    assert len(r["attempts"]) == 2


def test_narrow_retry_second_verify_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from token_reducer import patch_loop as pl

    f = tmp_path / "n.py"
    f.write_text("a = 1\n", encoding="utf-8")

    outcomes = [
        PatchResult(
            target_files=(str(f.resolve()),),
            diff_hunks="",
            rationale="1",
            patch_text=f"<<<< SEARCH {f.name}\na = 1\n==== REPLACE\na = 2\n>>>>",
            applicable=True,
        ),
        PatchResult(
            target_files=(str(f.resolve()),),
            diff_hunks="",
            rationale="2",
            patch_text=f"<<<< SEARCH {f.name}\na = 2\n==== REPLACE\na = 3\n>>>>",
            applicable=True,
        ),
    ]
    gen_iter = iter(outcomes)

    def next_gen(*a: object, **kw: object) -> PatchResult:
        return next(gen_iter)

    monkeypatch.setattr(pl, "generate_patch", next_gen)

    ver_calls: list[int] = []

    def ver_side(*a: object, **kw: object):
        ver_calls.append(1)
        if len(ver_calls) == 1:
            return VerificationOutcome(False, ("e",), (str(f.resolve()),))
        return VerificationOutcome(True, (), ())

    monkeypatch.setattr(pl, "run_verification", ver_side)

    monkeypatch.setattr(
        pl,
        "apply_patch_result",
        lambda *a, **k: ApplyOutcome(True, ("ok",), (str(f.resolve()),)),
    )

    st = MagicMock()
    st.query = "q"
    st.policy = MagicMock(
        patch_first=True,
        rewrite_query=True,
        verification_plan={"max_retries": 2, "lint_timeout_seconds": 5},
        boosted_sources=frozenset(),
        must_keep_symbol_tokens=frozenset(),
        effective_word_budget=200,
        effective_relevance_floor=0.1,
        code_invariant_level="standard",
        include_callers=True,
        include_callees=True,
        task_mode="debug",
        rerank_strategy="lexical_heavy",
        retry_on_low_score=True,
    )
    st.retrieval_plan = None
    st.repo_map = None
    st.selected = [
        Candidate(chunk_id=1, source=str(f), chunk_index=0, text="a=1", token_estimate=5, final_score=0.9)
    ]
    st.bullets = ["b"]

    def fts(*a: object, **kw: object):
        return [
            Candidate(chunk_id=2, source=str(f), chunk_index=0, text="a=2", token_estimate=5, final_score=0.5)
        ]

    monkeypatch.setattr(pl, "fts_retrieve", fts)

    r = run_closed_edit_loop(st, workspace_root=tmp_path, dry_run=True)
    assert r["verification"]["status"] == "passed"
    assert len(r["attempts"]) >= 2
