from __future__ import annotations

from token_reducer.repo_map import RepoFileRecord, RepoMap
from token_reducer.retrieval_plan import (
    build_retrieval_plan,
    intent_to_task_mode,
    plan_boosted_sources,
    plan_effective_fts_query,
    query_must_keep_tokens,
    query_suggests_code_exploration,
    query_suggests_dependency_tracing,
    query_suggests_downstream_impact,
    task_source_flags,
)


def _empty_repo() -> RepoMap:
    return RepoMap(
        files=(),
        test_sources=frozenset(),
        entry_sources=frozenset(),
        config_sources=frozenset(),
        service_sources=frozenset(),
        utility_sources=frozenset(),
        helper_sources=frozenset(),
        symbol_name_hints=frozenset(),
    )


def test_intent_to_task_mode_write_test_query() -> None:
    assert intent_to_task_mode("feature_add", "add pytest for login") == "write_test"


def test_plan_effective_fts_query_rewrite_strips_filler() -> None:
    q = plan_effective_fts_query("please fix the null pointer", "debug", rewrite=True)
    assert "please" not in q.lower()


def test_query_must_keep_tokens_extracts_identifiers() -> None:
    toks = query_must_keep_tokens("Why does UserService timeout?")
    assert "UserService" in toks


def test_query_suggests_code_exploration() -> None:
    assert query_suggests_code_exploration("How does UserService.connect work in service.py?") is True
    assert query_suggests_code_exploration("What is the meaning of life") is False


def test_task_source_flags_callers_for_debug() -> None:
    f = task_source_flags("debug", "")
    assert f["include_callers"] is True
    assert f["include_callees"] is True


def test_task_source_flags_callees_for_add_feature() -> None:
    assert task_source_flags("add_feature", "")["include_callees"] is True


def test_write_test_callers_only_with_dependency_hint() -> None:
    assert task_source_flags("write_test", "add pytest for login")["include_callers"] is False
    assert task_source_flags("write_test", "stack trace from caller")["include_callers"] is True


def test_refactor_callees_only_with_downstream_hint() -> None:
    assert task_source_flags("refactor", "rename variable x")["include_callees"] is False
    assert task_source_flags("refactor", "propagate rename to downstream consumers")["include_callees"] is True


def test_query_suggests_dependency_tracing() -> None:
    assert query_suggests_dependency_tracing("see traceback in logs") is True
    assert query_suggests_dependency_tracing("hello world") is False


def test_query_suggests_downstream_impact() -> None:
    assert query_suggests_downstream_impact("ripple effect on callers") is True
    assert query_suggests_downstream_impact("hello") is False


def test_build_retrieval_plan_caps_top_k_for_patch_tasks() -> None:
    rm = RepoMap(
        files=(
            RepoFileRecord("/p/tests/t.py", "test", True, False),
            RepoFileRecord("/p/main.py", "entry", False, True),
        ),
        test_sources=frozenset({"/p/tests/t.py"}),
        entry_sources=frozenset({"/p/main.py"}),
        config_sources=frozenset(),
        service_sources=frozenset(),
        utility_sources=frozenset(),
        helper_sources=frozenset(),
        symbol_name_hints=frozenset(),
    )
    plan = build_retrieval_plan(
        "refactor",
        "extract helper",
        rm,
        fts_k=20,
        top_k=8,
        retrieval_depth=2,
        rewrite_query=True,
        must_keep_tokens=frozenset({"Foo"}),
    )
    assert plan.effective_top_k <= 8
    assert plan.fts_cap >= 20
    assert plan.expand_neighbor_chunks is True


def test_plan_boosted_sources_prefers_tests_for_debug() -> None:
    rm = RepoMap(
        files=(
            RepoFileRecord("/p/tests/t.py", "test", True, False),
            RepoFileRecord("/p/main.py", "entry", False, True),
        ),
        test_sources=frozenset({"/p/tests/t.py"}),
        entry_sources=frozenset({"/p/main.py"}),
        config_sources=frozenset(),
        service_sources=frozenset(),
        utility_sources=frozenset(),
        helper_sources=frozenset(),
        symbol_name_hints=frozenset(),
    )
    b = plan_boosted_sources(rm, "debug")
    assert "/p/tests/t.py" in b


def test_build_retrieval_plan_empty_repo() -> None:
    plan = build_retrieval_plan(
        "explain",
        "overview",
        _empty_repo(),
        fts_k=10,
        top_k=5,
        retrieval_depth=1,
        rewrite_query=False,
        must_keep_tokens=frozenset(),
    )
    assert plan.fts_cap == 10
    assert plan.expand_neighbor_chunks is False
