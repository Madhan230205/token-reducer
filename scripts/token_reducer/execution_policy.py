"""Unified task + execution policy — drives retrieval weights, LSP, compression level, and verify hints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import should_skip_vector_for_hash
from .intent import IntentType
from .repo_map import RepoMap
from .retrieval_plan import (
    intent_to_task_mode,
    query_must_keep_tokens,
    query_suggests_code_exploration,
    task_source_flags,
)
from .verify_loop import build_verification_plan


def _pick_rerank_strategy(task_mode: str, use_vector: bool, patch_first: bool) -> str:
    if task_mode == "navigate":
        return "lexical_heavy"
    if task_mode == "debug":
        return "lexical_heavy"
    if patch_first and not use_vector:
        return "overlap_heavy"
    if use_vector and task_mode == "explain":
        return "semantic_heavy"
    if use_vector and task_mode == "refactor":
        return "balanced_hybrid"
    if use_vector and task_mode in ("add_feature", "write_test"):
        return "semantic_heavy"
    if patch_first and use_vector:
        return "overlap_heavy"
    return "default"


def _pick_invariant_level(task_mode: str) -> str:
    if task_mode in ("debug", "write_test"):
        return "strict"
    if task_mode in ("refactor", "add_feature"):
        return "standard"
    return "off"


@dataclass(frozen=True)
class ExecutionPolicy:
    """Control surface for a single context run (after initial lexical retrieval)."""

    task_mode: str
    use_fts: bool
    use_vector: bool
    vector_retrieval_path_if_disabled: str
    use_lsp: bool
    retrieval_depth: int
    lsp_symbol_fetch_limit: int
    effective_word_budget: int
    effective_relevance_floor: float
    rerank_strategy: str
    retry_on_low_score: bool
    rewrite_query: bool
    boosted_sources: frozenset[str]
    code_invariant_level: str
    must_keep_symbol_tokens: frozenset[str]
    include_tests: bool
    include_callers: bool
    include_callees: bool
    include_entry_points: bool
    patch_first: bool
    verification_plan: dict[str, Any] | None

    @property
    def code_invariant_compression(self) -> bool:
        """True when compressor should apply code preserve rules (not generic prose trim)."""
        return self.code_invariant_level != "off"


def derive_execution_policy(
    *,
    intent: IntentType,
    query: str,
    hybrid_mode: str,
    min_fts_hits: int,
    embedding_backend: str,
    adaptive_tier: str,
    fts_hit_count: int,
    word_budget: int,
    relevance_floor: float,
    repo_map: RepoMap | None = None,
    runtime_lint_cmds: dict[str, str] | None = None,
) -> ExecutionPolicy:
    """Derive policy from intent, query, corpus tier, FTS density, and optional repo map."""
    task_mode = intent_to_task_mode(intent, query)
    vector_path = "disabled"
    if hybrid_mode == "always":
        use_vector = True
    elif adaptive_tier == "fts_only":
        use_vector = False
        vector_path = "disabled_small_codebase"
    elif adaptive_tier == "fts_with_hash":
        use_vector = fts_hit_count < min_fts_hits
    else:
        use_vector = fts_hit_count < min_fts_hits

    if use_vector and embedding_backend == "hash" and should_skip_vector_for_hash():
        use_vector = False
        vector_path = "skipped_hash_backend"

    use_fts = True
    use_lsp = (task_mode != "explain") or query_suggests_code_exploration(query)

    retrieval_depth = 1
    retry = False
    rewrite = False
    if task_mode in ("debug", "write_test", "refactor", "add_feature"):
        retrieval_depth = 2
        retry = True
        rewrite = True

    lsp_limit = 3
    rf = relevance_floor
    wb = word_budget
    if intent == "bug_fix":
        rf = max(0.05, relevance_floor - 0.02)
        wb = int(word_budget * 1.05)
    elif intent == "navigation":
        rf = min(0.45, relevance_floor + 0.02)
        lsp_limit = 4
    elif intent == "explain_code":
        wb = int(word_budget * 1.02)
        if query_suggests_code_exploration(query):
            lsp_limit = 4
    elif intent == "feature_add":
        wb = int(word_budget * 1.03)

    must_keep = query_must_keep_tokens(query)
    flags = task_source_flags(task_mode, query)
    boosted: frozenset[str] = frozenset()
    if repo_map is not None:
        boosted = repo_map.boosted_sources_for_task(
            task_mode,
            must_keep,
            include_tests=flags["include_tests"],
            include_entry_points=flags["include_entry_points"],
            include_callers=flags["include_callers"],
            include_callees=flags["include_callees"],
            cap=72,
        )

    inv_level = _pick_invariant_level(task_mode)
    rs = _pick_rerank_strategy(task_mode, use_vector, flags["patch_first"])

    verify = build_verification_plan(
        task_mode,
        touched_files=None,
        runtime_lint_cmds=runtime_lint_cmds,
        query=query,
    )

    return ExecutionPolicy(
        task_mode=task_mode,
        use_fts=use_fts,
        use_vector=use_vector,
        vector_retrieval_path_if_disabled=vector_path,
        use_lsp=use_lsp,
        retrieval_depth=retrieval_depth,
        lsp_symbol_fetch_limit=max(1, min(10, lsp_limit)),
        effective_word_budget=max(16, wb),
        effective_relevance_floor=max(0.05, min(0.5, rf)),
        rerank_strategy=rs,
        retry_on_low_score=retry,
        rewrite_query=rewrite,
        boosted_sources=boosted,
        code_invariant_level=inv_level,
        must_keep_symbol_tokens=must_keep,
        include_tests=flags["include_tests"],
        include_callers=flags["include_callers"],
        include_callees=flags["include_callees"],
        include_entry_points=flags["include_entry_points"],
        patch_first=flags["patch_first"],
        verification_plan=verify,
    )
