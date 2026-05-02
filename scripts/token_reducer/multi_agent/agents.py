"""Four specialized agents — each owns a distinct phase of context construction.

These are **not** LLM personas: they are composable specialists with real,

non-overlapping responsibilities over :class:`~token_reducer.orchestrator.ContextRunState`.
Chunk-level “subagents” (filter/rank/fuse) remain under :mod:`token_reducer.subagents`.
"""

from __future__ import annotations

import os
from typing import Any

from ..audit_spine import apply_compression_slice, apply_retrieval_and_routing
from ..feedback import feedback_loop_adjustments_with_adaptive


class CostOptimizerAgent:
    """Tunes budgets before retrieval and enforces hard caps after compression (via :meth:`ContextPipelineOrchestrator.stage_cost_finalize`)."""

    name = "cost_optimizer"

    def run_prepare(self, orch: Any) -> dict[str, Any]:
        s = orch.state
        if s.feedback_loop_adj is None:
            s.feedback_loop_adj = feedback_loop_adjustments_with_adaptive(workspace_root=s.workspace_root)
        adj = s.feedback_loop_adj
        if not (getattr(s, "model_profile", None) or "").strip():
            s.model_profile = os.environ.get("TOKEN_REDUCER_MODEL", "sonnet").strip()
        orch.bootstrap_context_decision_d0()
        return {
            "feedback_retrieval_scale_mult": round(adj.retrieval_scale_mult, 4),
            "feedback_relevance_floor_delta": round(adj.relevance_floor_delta, 5),
            "model_profile": s.model_profile,
        }


class RetrieverAgent:
    """FTS/BM25 + optional vectors, fusion, re-rank, neighborhood expansion, and retrieval retry."""

    name = "retriever"

    def run(self, orch: Any) -> dict[str, Any]:
        orch.stage_retrieval()
        orch.stage_merge()
        orch.stage_scoring()
        orch.stage_neighborhood_expansion()
        orch.stage_finalize_context_intelligence()
        orch.maybe_retrieval_retry()
        if orch.state.retrieval_retry_done:
            orch.stage_finalize_context_intelligence()
        s = orch.state
        if s.audit_spine is not None:
            s.audit_spine = apply_retrieval_and_routing(
                s.audit_spine,
                scored_pool=s.scored_pool,
                fts_hits=s.fts_hits,
                vector_hits=s.vector_hits,
                route=s.execution_route,
                context_decision=s.context_decision,
            )
        return {
            "fts_hits": len(s.fts_hits),
            "vector_hits": len(s.vector_hits),
            "scored_pool": len(s.scored_pool),
            "vector_path": s.vector_retrieval_path,
        }


class ReasoningEnhancerAgent:
    """Delta-aware selection, intent-specialized chunk agents, then LSP definition snippets."""

    name = "reasoning_enhancer"

    def run(self, orch: Any) -> dict[str, Any]:
        orch.stage_final_selection()
        orch.stage_subagents()
        orch.stage_expansion()
        s = orch.state
        sub = s.subagent_debug if isinstance(s.subagent_debug, dict) else {}
        return {
            "selected_chunks": len(s.selected),
            "referenced_symbols": len(s.referenced_symbols),
            "subagent_profile": sub.get("profile") or s.subagent_profile_used,
            "subagent_chain": sub.get("subagents"),
        }


class CompressorAgent:
    """Model- and intent-aware compression plus structured plugin payload (summary + code_context)."""

    name = "compressor"

    def run(self, orch: Any) -> dict[str, Any]:
        orch.stage_compression()
        s = orch.state
        if s.audit_spine is not None:
            s.audit_spine = apply_compression_slice(
                s.audit_spine,
                selected=s.selected,
                omitted=s.omitted_redundant,
                bullets=s.bullets,
            )
        tok = sum(len(b.split()) for b in s.bullets)
        return {"bullet_count": len(s.bullets), "bullet_words_est": tok, "has_claude_context": s.claude_context is not None}
