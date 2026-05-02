"""Single orchestration path: retrieval → merge → rerank → selection → subagents → expansion → compression → packaging.

All context-building steps for a cache-miss query run through this module so the
pipeline is one ordered story, not ad-hoc calls scattered across callers.
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
from .context_pipeline import early_prune_candidates, expand_context_if_needed, rerank_chunks
from .context_explain import build_agent_trace, build_focus_line, chunk_transparency_rows
from .delta_context import (
    load_active_fingerprints,
    partition_redundant_candidates,
)
from .execution_policy import ExecutionPolicy, derive_execution_policy
from .feedback import read_feedback_source_boosts, read_strategy_prune_adjustments
from .intent import IntentType, detect_intent, structured_intent_to_dict
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


class ContextPipelineOrchestrator:
    """Runs the canonical stage order on a :class:`ContextRunState`."""

    __slots__ = ("state",)

    def __init__(self, state: ContextRunState) -> None:
        self.state = state

    def stage_retrieval(self) -> None:
        """Stage 1: BM25/FTS5 plus optional semantic (vector) retrieval."""
        s = self.state
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
        adj = read_strategy_prune_adjustments(workspace_root=s.workspace_root).get(strat.strategy_id, 0)
        if adj:
            strat = replace(strat, prune_k=max(6, min(18, strat.prune_k + adj)))
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
        s.scored_pool = rerank_chunks(
            list(s.merged_pool),
            s.query,
            structured,
            conn=s.conn,
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
        if not s.selected:
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
        out, meta = run_subagents(
            s.selected,
            s.query,
            si,
            {
                "model": model,
                "feedback_source_boost": boosts,
                "skip_fusion": bool(cs.get("skip_fusion", False)),
                "max_chunks": pk,
            },
        )
        s.selected = out
        s.subagent_debug = meta

    def stage_expansion(self) -> None:
        """Stage 5: LSP-backed definition expansion for the top surviving candidate."""
        s = self.state
        s.referenced_symbols = []
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
        if not s.selected and s.omitted_redundant:
            s.bullets = [
                f"(delta) {len(s.omitted_redundant)} chunk(s) status=omitted_redundant: "
                "already in active context (file hash/mtime unchanged). No new compressed text."
            ]
        else:
            structured = detect_intent(s.query)
            s.bullets = compress_context(
                s.selected,
                token_budget=int(structured["token_budget"]),
                query=s.query,
                relevance_floor=p.effective_relevance_floor if p else s.relevance_floor,
                code_invariant_level=p.code_invariant_level if p else "off",
                must_keep_tokens=p.must_keep_symbol_tokens if p else None,
                policy_word_budget=p.effective_word_budget if p else s.word_budget,
                compression_level=str(structured["compression_level"]),
            )
            s.bullets = expand_context_if_needed(
                s.bullets,
                structured,
                s.conn,
                s.scored_pool[:24],
            )
        s.claude_context = build_claude_plugin_payload(
            s.query,
            s.intent,
            s.selected,
            patch_first=bool(p and p.patch_first),
            context_strategy=s.context_strategy,
        )
        if s.policy and s.policy.patch_first and isinstance(s.claude_context, dict):
            s.claude_context = {
                **s.claude_context,
                "edit_style": "minimal_patch",
                "focus_paths": [c.source for c in s.selected[:8]],
            }

    def run_through_compression(self, *, debug: bool = False) -> dict[str, Any] | None:
        """Execute stages 1–6 in order (retrieval … compression), with optional retrieval retry."""
        self.stage_retrieval()
        self.stage_merge()
        self.stage_scoring()
        self.stage_neighborhood_expansion()
        s = self.state
        plan = s.retrieval_plan
        sparse_fts = len(s.fts_hits) < s.min_fts_hits
        weak = _weak_scored_pool(s.scored_pool)
        if (
            s.policy
            and s.policy.retry_on_low_score
            and plan
            and plan.second_pass_on_weak_pool
            and not s.retrieval_retry_done
            and (weak or sparse_fts)
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
        self.stage_final_selection()
        self.stage_subagents()
        self.stage_expansion()
        self.stage_compression()
        trace: dict[str, Any] | None = None
        if debug:
            trace = {
                "intent": structured_intent_to_dict(detect_intent(s.query)),
                "retrieved": len(s.fts_hits) + len(s.vector_hits),
                "reranked": len(s.scored_pool),
                "final_tokens": sum(len(b.split()) for b in s.bullets),
                "subagent_trace": getattr(s, "subagent_debug", None),
                "context_strategy": getattr(s, "context_strategy", None),
            }
            s.pipeline_debug = trace
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
