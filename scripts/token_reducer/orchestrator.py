"""Single orchestration path: four top-level specialists → packaging.

Specialists (:mod:`token_reducer.multi_agent`): **cost_optimizer** (feedback + model profile,

hard budget finalize), **retriever** (FTS/vector → merge → re-rank → neighborhood → retry),

**reasoning_enhancer** (selection, chunk subagents, LSP expansion), **compressor** (token

compression + plugin payload). Chunk-level subagents stay in :mod:`token_reducer.subagents`.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .chunker import function_call_positions
from .compressor import build_packet, compress_context
from .context_expansion import expand_context_candidates
from .context_explain import build_agent_trace, build_focus_line, chunk_transparency_rows
from .context_intelligence.heuristic import (
    build_decision_d0,
    build_decision_d1,
    intel_retry_allowed,
)
from .context_intelligence.models import ContextDecision
from .context_intelligence.strategy_nudge import apply_task_shape_nudge
from .context_intelligence.telemetry import build_intel_debug_payload
from .context_pipeline import early_prune_candidates, expand_context_if_needed, rerank_chunks
from .delta_context import (
    load_active_fingerprints,
    partition_redundant_candidates,
)
from .execution_policy import ExecutionPolicy, derive_execution_policy
from .execution_route import ExecutionRoute, RoutingPlan
from .feedback import (
    FeedbackLoopAdjustments,
    adaptive_prune_integer_delta,
    adaptive_skill_prior_nudge,
    build_adaptation_brief,
    feedback_loop_adjustments_with_adaptive,
    read_feedback_source_boosts,
    read_strategy_prune_adjustments,
)
from .intent import IntentType, detect_intent, intent_score_map, structured_intent_to_dict
from .lsp_client import HeadlessLSPClient
from .models import Candidate, ContextPacket, OmittedRedundantEntry
from .plugin_output import build_claude_plugin_payload
from .plugin_settings import TokenReducerRuntimeConfig
from .repo_map import RepoMap, build_repo_map
from .retrieval_boost import apply_source_boost
from .retrieval_plan import (
    RetrievalPlan,
    build_retrieval_plan,
    intent_to_task_mode,
    plan_effective_fts_query,
)
from .retriever import fts_retrieve, infer_retrieval_tier, rerank_candidates, vector_retrieve
from .subagents.router import run_subagents


def _weak_scored_pool(pool: list[Candidate]) -> bool:
    """True when retrieval likely missed — triggers a narrow second FTS pass."""
    if not pool:
        return True
    top = float(pool[0].final_score)
    if len(pool) < 3 and top < 0.22:
        return True
    return top < 0.13


@dataclass
class ContextRunState:
    """Mutable state passed sequentially through orchestration stages."""

    conn: sqlite3.Connection
    db_path: Path
    query: str
    intent: IntentType
    runtime: TokenReducerRuntimeConfig
    memory_blob: dict
    session_id: str
    top_k: int
    fts_k: int
    vector_k: int
    min_fts_hits: int
    hybrid_mode: str
    embedding_backend: str
    embedding_model: str | None
    dimensions: int
    word_budget: int
    relevance_floor: float
    workspace_root: Path | None
    execution_route: ExecutionRoute | None = None
    routing_plan: RoutingPlan | None = None
    subagent_profile_used: str | None = None
    feedback_loop_adj: FeedbackLoopAdjustments | None = None

    policy: ExecutionPolicy | None = None
    repo_map: RepoMap | None = None
    retrieval_plan: RetrievalPlan | None = None
    retrieval_retry_done: bool = False

    fts_hits: list[Candidate] = field(default_factory=list)
    vector_hits: list[Candidate] = field(default_factory=list)
    vector_backend_used: str = "disabled"
    vector_model_used: str | None = None
    vector_retrieval_path: str = "disabled"
    merged_pool: list[Candidate] = field(default_factory=list)
    scored_pool: list[Candidate] = field(default_factory=list)
    selected: list[Candidate] = field(default_factory=list)
    omitted_redundant: list[OmittedRedundantEntry] = field(default_factory=list)
    referenced_symbols: list[dict] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    claude_context: dict | None = None
    agent_result: dict[str, Any] | None = None
    pipeline_debug: dict[str, Any] | None = None
    model_profile: str | None = None
    subagent_debug: dict[str, Any] | None = None
    context_strategy: dict[str, Any] | None = None
    context_decision_d0: ContextDecision | None = None
    context_decision: ContextDecision | None = None
    strategy_baseline_merge_cap: int | None = None
    strategy_baseline_prune_k: int | None = None


class ContextPipelineOrchestrator:
    """Runs the canonical stage order on a :class:`ContextRunState`."""

    __slots__ = ("state",)

    def __init__(self, state: ContextRunState) -> None:
        self.state = state

    def bootstrap_context_decision_d0(self) -> None:
        """Build provisional D₀ once routing + feedback prep have populated execution tier."""
        s = self.state
        tier = s.execution_route.tier if s.execution_route else None
        si = structured_intent_to_dict(detect_intent(s.query))
        thr_raw = os.environ.get("TOKEN_REDUCER_CONTEXT_INTEL_THRESHOLD")
        thr: float | None = None
        if thr_raw:
            try:
                thr = float(thr_raw)
            except ValueError:
                thr = None
        s.context_decision_d0 = build_decision_d0(
            s.query, si, execution_tier=tier, confidence_threshold=thr
        )

    def stage_finalize_context_intelligence(self) -> None:
        """Freeze D₁ using corpus signals (§6); optional §7 telemetry."""
        s = self.state
        d0 = s.context_decision_d0
        if d0 is None:
            return
        scores = intent_score_map(s.query)
        weak_legacy = _weak_scored_pool(s.scored_pool)
        s.context_decision = build_decision_d1(
            d0,
            query=s.query,
            scores=scores,
            fts_hits=len(s.fts_hits),
            vector_hits=len(s.vector_hits),
            min_fts_hits=s.min_fts_hits,
            scored_pool=s.scored_pool,
            weak_pool_legacy=weak_legacy,
        )
        if os.environ.get("TOKEN_REDUCER_CONTEXT_INTEL_DEBUG"):
            merged = dict(s.pipeline_debug or {})
            merged["context_intelligence"] = build_intel_debug_payload(s.context_decision)
            s.pipeline_debug = merged

    def stage_retrieval(self) -> None:
        """Stage 1: BM25/FTS5 plus optional semantic (vector) retrieval."""
        s = self.state
        route = s.execution_route
        if s.feedback_loop_adj is None:
            s.feedback_loop_adj = feedback_loop_adjustments_with_adaptive(
                workspace_root=s.workspace_root,
            )
        adj = s.feedback_loop_adj
        route_sc = route.retrieval_scale if route is not None else 1.0
        sc = max(0.5, min(1.15, route_sc * adj.retrieval_scale_mult))
        s.top_k = max(6, int(round(s.top_k * sc)))
        s.fts_k = max(8, int(round(s.fts_k * sc)))
        if s.vector_k:
            s.vector_k = max(4, int(round(s.vector_k * sc)))
        if s.repo_map is None:
            s.repo_map = build_repo_map(s.conn)
        tm = intent_to_task_mode(s.intent, s.query)
        fts_query_text = plan_effective_fts_query(s.query, tm, rewrite=False)
        s.fts_hits = fts_retrieve(s.conn, fts_query_text, limit=s.fts_k)
        s.vector_hits = []
        s.vector_backend_used = "disabled"
        s.vector_model_used = None
        s.vector_retrieval_path = "disabled"

        adaptive_tier = infer_retrieval_tier(s.conn)
        s.policy = derive_execution_policy(
            intent=s.intent,
            query=s.query,
            hybrid_mode=s.hybrid_mode,
            min_fts_hits=s.min_fts_hits,
            embedding_backend=s.embedding_backend,
            adaptive_tier=adaptive_tier,
            fts_hit_count=len(s.fts_hits),
            word_budget=s.word_budget,
            relevance_floor=s.relevance_floor,
            repo_map=s.repo_map,
            runtime_lint_cmds=s.runtime.shadow_linter_cmds,
        )

        if not s.policy.use_vector:
            s.vector_retrieval_path = s.policy.vector_retrieval_path_if_disabled

        if s.policy.use_vector:
            s.vector_hits, s.vector_backend_used, s.vector_model_used, s.vector_retrieval_path = (
                vector_retrieve(
                    conn=s.conn,
                    db_path=s.db_path,
                    query=s.query,
                    limit=s.vector_k,
                    dimensions=s.dimensions,
                    embedding_backend=s.embedding_backend,
                    embedding_model=s.embedding_model,
                )
            )

        s.retrieval_plan = build_retrieval_plan(
            s.policy.task_mode,
            s.query,
            s.repo_map,
            s.fts_k,
            s.top_k,
            s.policy.retrieval_depth,
            s.policy.rewrite_query,
            s.policy.must_keep_symbol_tokens,
        )
        if s.retrieval_plan.fts_cap > s.fts_k:
            s.fts_hits = fts_retrieve(
                s.conn,
                s.retrieval_plan.effective_fts_query,
                s.retrieval_plan.fts_cap,
            )
            s.policy = derive_execution_policy(
                intent=s.intent,
                query=s.query,
                hybrid_mode=s.hybrid_mode,
                min_fts_hits=s.min_fts_hits,
                embedding_backend=s.embedding_backend,
                adaptive_tier=adaptive_tier,
                fts_hit_count=len(s.fts_hits),
                word_budget=s.word_budget,
                relevance_floor=s.relevance_floor,
                repo_map=s.repo_map,
                runtime_lint_cmds=s.runtime.shadow_linter_cmds,
            )
            if not s.policy.use_vector:
                s.vector_hits = []
                s.vector_backend_used = "disabled"
                s.vector_model_used = None
                s.vector_retrieval_path = s.policy.vector_retrieval_path_if_disabled
            else:
                s.vector_hits, s.vector_backend_used, s.vector_model_used, s.vector_retrieval_path = (
                    vector_retrieve(
                        conn=s.conn,
                        db_path=s.db_path,
                        query=s.query,
                        limit=s.vector_k,
                        dimensions=s.dimensions,
                        embedding_backend=s.embedding_backend,
                        embedding_model=s.embedding_model,
                    )
                )

        from .context_strategy import map_query_to_strategy

        si = structured_intent_to_dict(detect_intent(s.query))
        strat = map_query_to_strategy(
            s.query,
            si,
            adaptive_tier,
            use_vector=bool(s.policy and s.policy.use_vector),
        )
        baseline_m = int(strat.merge_cap)
        baseline_p = int(strat.prune_k)
        s.strategy_baseline_merge_cap = baseline_m
        s.strategy_baseline_prune_k = baseline_p
        adj_raw = read_strategy_prune_adjustments(workspace_root=s.workspace_root).get(
            strat.strategy_id, 0
        )
        adj_val = max(
            -1,
            min(
                1,
                int(adj_raw) + adaptive_prune_integer_delta(s.workspace_root, strat.strategy_id),
            ),
        )
        merged_m, merged_p = baseline_m, baseline_p
        if s.context_decision_d0 is not None:
            merged_m, merged_p = apply_task_shape_nudge(
                baseline_m, baseline_p, s.context_decision_d0.task_shape
            )
        prune_k = merged_p
        if adj_val:
            prune_k = max(6, min(18, prune_k + adj_val))
        strat = replace(strat, merge_cap=merged_m, prune_k=prune_k)
        s.context_strategy = strat.to_dict()

    def stage_merge(self) -> None:
        """Stage 2: fuse lexical + vector hit lists (RRF or weighted merge)."""
        s = self.state
        strat = s.policy.rerank_strategy if s.policy else "default"
        si = detect_intent(s.query)
        effective_top_k = min(s.top_k, max(6, int(si["k"])))
        _, pool = rerank_candidates(
            query=s.query,
            fts_hits=s.fts_hits,
            vector_hits=s.vector_hits,
            top_k=effective_top_k,
            strategy=strat,
        )
        cs = s.context_strategy or {}
        s.merged_pool = early_prune_candidates(
            list(pool),
            merge_cap=int(cs.get("merge_cap", 50)),
            prune_k=int(cs.get("prune_k", 15)),
        )

    def stage_scoring(self) -> None:
        """Stage 3: post-merge re-rank (semantic + keyword + recency + structural blend)."""
        s = self.state
        structured = detect_intent(s.query)
        rp = "default"
        rw = None
        if s.context_decision_d0 is not None:
            rp = s.context_decision_d0.ranking_profile or "default"
            rw = s.context_decision_d0.ranking_weights
        s.scored_pool = rerank_chunks(
            list(s.merged_pool),
            s.query,
            structured,
            conn=s.conn,
            ranking_profile=rp,
            ranking_weights=rw,
        )
        if s.policy and s.policy.boosted_sources:
            apply_source_boost(
                s.scored_pool,
                s.policy.boosted_sources,
                patch_first=s.policy.patch_first,
            )

    def stage_neighborhood_expansion(self) -> None:
        """Inject adjacent chunks + light caller/callee fanout before final selection."""
        s = self.state
        if s.execution_route and s.execution_route.tier == "simple":
            return
        if s.context_strategy and s.context_strategy.get("skip_neighborhood"):
            return
        plan = s.retrieval_plan
        p = s.policy
        if not plan or not plan.expand_neighbor_chunks or not p:
            return
        if not p.include_callers and not p.include_callees:
            return
        extra = expand_context_candidates(
            s.conn,
            s.scored_pool,
            s.repo_map,
            p.must_keep_symbol_tokens,
            include_callers=p.include_callers,
            include_callees=p.include_callees,
            max_extra=22,
        )
        if not extra:
            return
        existing = {c.chunk_id for c in s.scored_pool}
        for c in extra:
            if c.chunk_id in existing:
                continue
            existing.add(c.chunk_id)
            s.scored_pool.append(c)
        s.scored_pool.sort(key=lambda x: x.final_score, reverse=True)
        if p.boosted_sources:
            apply_source_boost(
                s.scored_pool,
                p.boosted_sources,
                patch_first=p.patch_first,
            )

    def maybe_retrieval_retry(self) -> None:
        """Second FTS pass when the pool is sparse or top scores are weak."""
        s = self.state
        plan = s.retrieval_plan
        sparse_fts = len(s.fts_hits) < s.min_fts_hits
        weak = _weak_scored_pool(s.scored_pool)
        intel_ok = True
        if s.context_decision is not None:
            intel_ok = intel_retry_allowed(s.context_decision, weak)
        if (
            s.policy
            and s.policy.retry_on_low_score
            and plan
            and plan.second_pass_on_weak_pool
            and not s.retrieval_retry_done
            and (weak or sparse_fts)
            and intel_ok
        ):
            s.retrieval_retry_done = True
            tm = intent_to_task_mode(s.intent, s.query)
            q2 = plan_effective_fts_query(s.query, tm, rewrite=plan.rewrite_on_second_pass)
            extra = fts_retrieve(s.conn, q2, limit=min(s.fts_k + 14, 96))
            seen = {c.chunk_id for c in s.fts_hits}
            for c in extra:
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    s.fts_hits.append(c)
            self.stage_merge()
            self.stage_scoring()
            self.stage_neighborhood_expansion()

    def stage_final_selection(self) -> None:
        """Stage 4: take top-k, then apply session delta (omit unchanged redundant chunks)."""
        s = self.state
        cap = s.retrieval_plan.effective_top_k if s.retrieval_plan else s.top_k
        pre = s.scored_pool[:cap]
        active_fps = load_active_fingerprints(s.memory_blob, s.session_id)
        s.selected, s.omitted_redundant = partition_redundant_candidates(s.conn, active_fps, pre)

    def stage_subagents(self) -> None:
        """Pre-compression deterministic passes (filter / dedup / fuse / budget)."""
        s = self.state
        if s.execution_route and s.execution_route.skip_subagents:
            n = len(s.selected)
            tok = sum(int(c.token_estimate) for c in s.selected)
            s.subagent_profile_used = "none"
            s.subagent_debug = {
                "subagents": [],
                "skipped": "execution_route",
                "chunks_before": n,
                "chunks_after": n,
                "tokens_before": tok,
                "tokens_after": tok,
                "tokens_saved": 0,
            }
            return
        if not s.selected:
            s.subagent_profile_used = None
            s.subagent_debug = {
                "subagents": [],
                "chunks_before": 0,
                "chunks_after": 0,
                "tokens_before": 0,
                "tokens_after": 0,
                "tokens_saved": 0,
            }
            return
        si = detect_intent(s.query)
        model = (s.model_profile or os.environ.get("TOKEN_REDUCER_MODEL", "sonnet")).strip()
        boosts = read_feedback_source_boosts()
        cs = s.context_strategy or {}
        pk = max(4, min(18, int(cs.get("prune_k", 15))))
        profile = "full"
        rp = s.routing_plan
        if rp is not None and rp.subagent_profile != "none":
            profile = rp.subagent_profile
        if profile == "full":
            sid = str(cs.get("strategy_id") or "")
            if sid:
                adj = max(
                    -1,
                    min(
                        1,
                        int(read_strategy_prune_adjustments(workspace_root=s.workspace_root).get(sid, 0))
                        + adaptive_prune_integer_delta(s.workspace_root, sid),
                    ),
                )
                if adj <= -1:
                    profile = "lean"
        skill_id = s.execution_route.skill_id if s.execution_route else None
        out, meta = run_subagents(
            s.selected,
            s.query,
            si,
            {
                "model": model,
                "feedback_source_boost": boosts,
                "skip_fusion": bool(cs.get("skip_fusion", False)),
                "max_chunks": pk,
                "adaptive_skill_prior_nudge": adaptive_skill_prior_nudge(s.workspace_root, skill_id),
            },
            profile=profile,
        )
        s.selected = out
        s.subagent_profile_used = profile
        s.subagent_debug = meta

    def stage_expansion(self) -> None:
        """Stage 5: LSP-backed definition expansion for the top surviving candidate."""
        s = self.state
        s.referenced_symbols = []
        if s.execution_route and s.execution_route.skip_lsp:
            return
        if not s.selected:
            return
        if s.policy and not s.policy.use_lsp:
            return
        top = s.selected[0]
        ext = Path(top.source).suffix.lower()
        cmd = s.runtime.lsp_servers.get(ext)
        if not cmd or not shutil.which(cmd[0]):
            return
        root = s.workspace_root or Path(top.source).resolve().parent
        src_path = Path(top.source).resolve()
        client: HeadlessLSPClient | None = None
        try:
            client = HeadlessLSPClient(cmd, root)
            init_resp = client.initialize()
            if init_resp and "error" not in init_resp:
                client.open_file(src_path, top.text, ext)
                limit = s.policy.lsp_symbol_fetch_limit if s.policy else 3
                for name, line, col in function_call_positions(top.text, limit=limit):
                    for snip in client.definition_snippet(src_path, line, col):
                        s.referenced_symbols.append(
                            {
                                "symbol": name,
                                "from_chunk": top.source,
                                **snip,
                            }
                        )
        except (
            FileNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
            json.JSONDecodeError,
            TypeError,
            KeyError,
        ):
            pass
        finally:
            if client is not None:
                with contextlib.suppress(OSError):
                    client.shutdown()

    def stage_compression(self) -> None:
        """Stage 6: token-budget compression then optional bullet expansion."""
        s = self.state
        p = s.policy
        adj = s.feedback_loop_adj or feedback_loop_adjustments_with_adaptive(workspace_root=s.workspace_root)
        model_for = (s.model_profile or os.environ.get("TOKEN_REDUCER_MODEL", "sonnet")).strip()
        comp_level_for_brief: str | None = None
        if not s.selected and s.omitted_redundant:
            s.bullets = [
                f"(delta) {len(s.omitted_redundant)} chunk(s) status=omitted_redundant: "
                "already in active context (file hash/mtime unchanged). No new compressed text."
            ]
        else:
            structured = detect_intent(s.query)
            route = s.execution_route
            rp = s.routing_plan
            tbudget = int(structured["token_budget"])
            if rp is not None:
                tbudget = min(tbudget, rp.output_token_cap)
            pre_tok = sum(int(c.token_estimate) for c in s.selected)
            heavy_compress = True if route is None else pre_tok >= route.reducer_token_threshold
            comp_level = str(structured["compression_level"])
            if not heavy_compress:
                comp_level = "low"
            comp_level_for_brief = comp_level
            base_rf = p.effective_relevance_floor if p else s.relevance_floor
            relevance_floor = max(0.05, min(0.5, base_rf + adj.relevance_floor_delta))
            cd = s.context_decision
            if cd is not None and abs(cd.compression_delta) > 1e-9:
                shift = max(-0.06, min(0.06, cd.compression_delta * 0.04))
                relevance_floor = max(0.05, min(0.5, relevance_floor + shift))
            s.bullets = compress_context(
                s.selected,
                token_budget=tbudget,
                query=s.query,
                relevance_floor=relevance_floor,
                code_invariant_level=p.code_invariant_level if p else "off",
                must_keep_tokens=p.must_keep_symbol_tokens if p else None,
                policy_word_budget=p.effective_word_budget if p else s.word_budget,
                compression_level=comp_level,
                legacy_intent=str(s.intent),
                model_profile=model_for,
            )
            if heavy_compress:
                s.bullets = expand_context_if_needed(
                    s.bullets,
                    structured,
                    s.conn,
                    s.scored_pool[:24],
                )
        show_adapt = os.environ.get("TOKEN_REDUCER_ADAPTATION_BRIEF")
        show_route_dbg = os.environ.get("TOKEN_REDUCER_ROUTING_DEBUG")
        adapt_lines: list[str] = []
        if show_adapt or show_route_dbg:
            adapt_lines = build_adaptation_brief(
                adj=adj,
                model=model_for,
                feedback_source_boosts=read_feedback_source_boosts(),
                subagent_meta=s.subagent_debug,
                compression_level_effective=comp_level_for_brief,
                execution_tier=s.execution_route.tier if s.execution_route else None,
            )
        routing_debug: dict[str, object] | None = None
        if show_route_dbg:
            routing_debug = {
                "tier": s.execution_route.tier if s.execution_route else None,
                "output_token_cap": s.routing_plan.output_token_cap if s.routing_plan else None,
                "subagents": (s.subagent_debug or {}).get("subagents"),
                "profile": (s.subagent_debug or {}).get("profile") or s.subagent_profile_used,
                "adaptation_brief": adapt_lines,
            }
        payload_adapt = adapt_lines if show_adapt else None
        s.claude_context = build_claude_plugin_payload(
            s.query,
            s.intent,
            s.selected,
            patch_first=bool(p and p.patch_first),
            context_strategy=s.context_strategy,
            routing_debug=routing_debug,
            adaptation_brief=payload_adapt,
        )
        if s.policy and s.policy.patch_first and isinstance(s.claude_context, dict):
            s.claude_context = {
                **s.claude_context,
                "edit_style": "minimal_patch",
                "focus_paths": [c.source for c in s.selected[:8]],
            }

    def stage_cost_finalize(self, prompt: str) -> dict[str, Any]:
        """Hard-enforce output bullet budget (CostOptimizerAgent finalize phase)."""
        from . import context_pipeline as _cp
        from .intent import detect_intent, structured_intent_to_dict
        from .subagents.policy import effective_chunk_token_budget
        from .token_intelligence import refine_output_bullets

        s = self.state
        structured = structured_intent_to_dict(detect_intent(prompt))
        model = (s.model_profile or os.environ.get("TOKEN_REDUCER_MODEL", "sonnet")).strip()
        out_budget = effective_chunk_token_budget(int(structured["token_budget"]), model)
        rp = s.routing_plan
        if rp is not None:
            out_budget = min(out_budget, rp.output_token_cap)
        s.bullets = refine_output_bullets(s.bullets, prompt)
        s.bullets = _cp.ensure_bullets_within_token_budget(s.bullets, out_budget)
        final_tok = _cp._final_bullet_token_estimate(s.bullets)
        assert final_tok <= out_budget, f"final_tokens {final_tok} exceed budget {out_budget}"
        return {"output_token_budget": out_budget, "final_tokens": final_tok}

    def run_through_compression(self, *, debug: bool = False) -> dict[str, Any] | None:
        """Execute specialists: cost prepare → retrieve → reasoning → compress."""
        from .multi_agent.runner import run_multi_agent_compression_phases

        ma = run_multi_agent_compression_phases(self)
        trace: dict[str, Any] | None = None
        if debug:
            s = self.state
            prev = dict(s.pipeline_debug or {})
            trace = {
                "intent": structured_intent_to_dict(detect_intent(s.query)),
                "retrieved": len(s.fts_hits) + len(s.vector_hits),
                "reranked": len(s.scored_pool),
                "final_tokens": sum(len(b.split()) for b in s.bullets),
                "subagent_trace": getattr(s, "subagent_debug", None),
                "context_strategy": getattr(s, "context_strategy", None),
                "multi_agent": ma,
            }
            merged = {**prev, **trace}
            if "context_intelligence" in prev:
                merged["context_intelligence"] = prev["context_intelligence"]
            s.pipeline_debug = merged
        return trace


def build_packet_from_state(
    state: ContextRunState,
    *,
    retrieval_mode: str,
    hybrid_mode: str,
    active_sig: str,
) -> ContextPacket:
    """Stage 7: assemble the context packet for the session/UI."""
    cs = getattr(state, "context_strategy", None) or {}
    sid = cs.get("strategy_id") if isinstance(cs, dict) else None
    sid_str = str(sid) if sid else None
    focus = build_focus_line(
        state.query,
        state.intent,
        state.selected,
        strategy_id=sid_str,
    )
    trace = build_agent_trace(state)
    trans = chunk_transparency_rows(state.selected)
    return build_packet(
        query=state.query,
        selected=state.selected,
        candidate_pool=state.scored_pool,
        bullets=state.bullets,
        fts_hit_count=len(state.fts_hits),
        vector_hit_count=len(state.vector_hits),
        hybrid_mode=hybrid_mode,
        retrieval_mode=retrieval_mode,
        vector_backend_used=state.vector_backend_used,
        vector_model_used=state.vector_model_used,
        vector_retrieval_path=state.vector_retrieval_path,
        omitted_redundant=state.omitted_redundant,
        active_context_signature=active_sig,
        referenced_symbols=state.referenced_symbols or None,
        claude_context=state.claude_context,
        task_mode=state.policy.task_mode if state.policy else None,
        patch_first=state.policy.patch_first if state.policy else False,
        verification_plan=state.policy.verification_plan if state.policy else None,
        agent_result=getattr(state, "agent_result", None),
        focus_line=focus,
        agent_trace=trace,
        chunk_transparency=trans,
    )
