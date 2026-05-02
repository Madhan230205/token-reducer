from __future__ import annotations

from token_reducer.execution_policy import derive_execution_policy


def test_vector_always_hybrid_mode() -> None:
    p = derive_execution_policy(
        intent="explain_code",
        query="how does indexing work",
        hybrid_mode="always",
        min_fts_hits=99,
        embedding_backend="ml",
        adaptive_tier="fts_only",
        fts_hit_count=100,
        word_budget=400,
        relevance_floor=0.15,
    )
    assert p.use_vector is True


def test_vector_disabled_small_codebase() -> None:
    p = derive_execution_policy(
        intent="explain_code",
        query="explain",
        hybrid_mode="fallback",
        min_fts_hits=3,
        embedding_backend="ml",
        adaptive_tier="fts_only",
        fts_hit_count=0,
        word_budget=400,
        relevance_floor=0.15,
    )
    assert p.use_vector is False
    assert p.vector_retrieval_path_if_disabled == "disabled_small_codebase"


def test_vector_sparse_fts_full_hybrid() -> None:
    p = derive_execution_policy(
        intent="explain_code",
        query="explain",
        hybrid_mode="fallback",
        min_fts_hits=5,
        embedding_backend="ml",
        adaptive_tier="full_hybrid",
        fts_hit_count=1,
        word_budget=400,
        relevance_floor=0.15,
    )
    assert p.use_vector is True


def test_vector_skipped_when_dense_fts_hits() -> None:
    p = derive_execution_policy(
        intent="explain_code",
        query="explain",
        hybrid_mode="fallback",
        min_fts_hits=3,
        embedding_backend="ml",
        adaptive_tier="full_hybrid",
        fts_hit_count=10,
        word_budget=400,
        relevance_floor=0.15,
    )
    assert p.use_vector is False
    assert p.vector_retrieval_path_if_disabled == "disabled"


def test_navigation_raises_lsp_limit() -> None:
    p = derive_execution_policy(
        intent="navigation",
        query="where is the handler",
        hybrid_mode="fallback",
        min_fts_hits=3,
        embedding_backend="ml",
        adaptive_tier="full_hybrid",
        fts_hit_count=10,
        word_budget=400,
        relevance_floor=0.15,
    )
    assert p.lsp_symbol_fetch_limit == 4


def test_bug_fix_lowers_relevance_floor() -> None:
    base = 0.2
    p = derive_execution_policy(
        intent="bug_fix",
        query="crash on timeout",
        hybrid_mode="fallback",
        min_fts_hits=3,
        embedding_backend="ml",
        adaptive_tier="full_hybrid",
        fts_hit_count=10,
        word_budget=400,
        relevance_floor=base,
    )
    assert p.effective_relevance_floor < base


def test_explain_mode_no_retrieval_retry() -> None:
    p = derive_execution_policy(
        intent="explain_code",
        query="what is a chunk",
        hybrid_mode="fallback",
        min_fts_hits=3,
        embedding_backend="ml",
        adaptive_tier="full_hybrid",
        fts_hit_count=10,
        word_budget=400,
        relevance_floor=0.15,
    )
    assert p.retry_on_low_score is False
    assert p.retrieval_depth == 1
    assert p.patch_first is False


def test_refactor_enables_retry_patch_and_invariants() -> None:
    p = derive_execution_policy(
        intent="refactor",
        query="extract helper from FooService",
        hybrid_mode="fallback",
        min_fts_hits=3,
        embedding_backend="ml",
        adaptive_tier="full_hybrid",
        fts_hit_count=10,
        word_budget=400,
        relevance_floor=0.15,
    )
    assert p.retry_on_low_score is True
    assert p.rewrite_query is True
    assert p.retrieval_depth >= 2
    assert p.patch_first is True
    assert p.code_invariant_level == "standard"
    assert p.code_invariant_compression is True
    assert p.verification_plan is not None
    assert p.verification_plan.get("preferred_output") == "patch"


def test_refactor_fts_only_overlap_rerank() -> None:
    p = derive_execution_policy(
        intent="refactor",
        query="extract helper",
        hybrid_mode="fallback",
        min_fts_hits=3,
        embedding_backend="ml",
        adaptive_tier="fts_only",
        fts_hit_count=50,
        word_budget=400,
        relevance_floor=0.15,
    )
    assert p.use_vector is False
    assert p.patch_first is True
    assert p.rerank_strategy == "overlap_heavy"


def test_explain_fts_only_default_rerank() -> None:
    p = derive_execution_policy(
        intent="explain_code",
        query="what is a chunk",
        hybrid_mode="fallback",
        min_fts_hits=3,
        embedding_backend="ml",
        adaptive_tier="fts_only",
        fts_hit_count=50,
        word_budget=400,
        relevance_floor=0.15,
    )
    assert p.use_vector is False
    assert p.rerank_strategy == "default"


def test_refactor_full_hybrid_balanced_rerank_when_vector_on() -> None:
    p = derive_execution_policy(
        intent="refactor",
        query="extract helper",
        hybrid_mode="fallback",
        min_fts_hits=5,
        embedding_backend="ml",
        adaptive_tier="full_hybrid",
        fts_hit_count=1,
        word_budget=400,
        relevance_floor=0.15,
    )
    assert p.use_vector is True
    assert p.rerank_strategy == "balanced_hybrid"


def test_debug_task_uses_lexical_rerank_and_callers() -> None:
    p = derive_execution_policy(
        intent="bug_fix",
        query="stack trace in handler.py",
        hybrid_mode="fallback",
        min_fts_hits=3,
        embedding_backend="ml",
        adaptive_tier="full_hybrid",
        fts_hit_count=10,
        word_budget=400,
        relevance_floor=0.15,
    )
    assert p.rerank_strategy == "lexical_heavy"
    assert p.include_callers is True
    assert p.include_callees is True
    assert p.code_invariant_level == "strict"


def test_explain_plain_query_skips_lsp() -> None:
    p = derive_execution_policy(
        intent="explain_code",
        query="what is a chunk in the index",
        hybrid_mode="fallback",
        min_fts_hits=3,
        embedding_backend="ml",
        adaptive_tier="full_hybrid",
        fts_hit_count=10,
        word_budget=400,
        relevance_floor=0.15,
    )
    assert p.use_lsp is False


def test_explain_codeish_query_enables_lsp() -> None:
    p = derive_execution_policy(
        intent="explain_code",
        query="How does UserService.connect() work in service.py?",
        hybrid_mode="fallback",
        min_fts_hits=3,
        embedding_backend="ml",
        adaptive_tier="full_hybrid",
        fts_hit_count=10,
        word_budget=400,
        relevance_floor=0.15,
    )
    assert p.use_lsp is True
