"""Single orchestration path: retrieval → merge → scoring → selection → expansion → compression → packaging.

All context-building steps for a cache-miss query run through this module so the
pipeline is one ordered story, not ad-hoc calls scattered across callers.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import function_call_positions
from .compressor import build_packet, compress_candidates
from .config import should_skip_vector_for_hash
from .delta_context import (
    load_active_fingerprints,
    partition_redundant_candidates,
)
from .intent import IntentType
from .lsp_client import HeadlessLSPClient
from .models import Candidate, ContextPacket, OmittedRedundantEntry
from .plugin_output import build_claude_plugin_payload
from .plugin_settings import TokenReducerRuntimeConfig
from .ranking import apply_claude_intent_rerank
from .retriever import fts_retrieve, infer_retrieval_tier, rerank_candidates, vector_retrieve


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


class ContextPipelineOrchestrator:
    """Runs the canonical stage order on a :class:`ContextRunState`."""

    __slots__ = ("state",)

    def __init__(self, state: ContextRunState) -> None:
        self.state = state

    def stage_retrieval(self) -> None:
        """Stage 1: BM25/FTS5 plus optional semantic (vector) retrieval."""
        s = self.state
        s.fts_hits = fts_retrieve(s.conn, s.query, limit=s.fts_k)
        s.vector_hits = []
        s.vector_backend_used = "disabled"
        s.vector_model_used = None
        s.vector_retrieval_path = "disabled"

        adaptive_tier = infer_retrieval_tier(s.conn)
        if s.hybrid_mode == "always":
            use_vector = True
        elif adaptive_tier == "fts_only":
            use_vector = False
            s.vector_retrieval_path = "disabled_small_codebase"
        elif adaptive_tier == "fts_with_hash":
            use_vector = len(s.fts_hits) < s.min_fts_hits
        else:
            use_vector = len(s.fts_hits) < s.min_fts_hits

        if use_vector and s.embedding_backend == "hash" and should_skip_vector_for_hash():
            use_vector = False
            s.vector_retrieval_path = "skipped_hash_backend"

        if use_vector:
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

    def stage_merge(self) -> None:
        """Stage 2: fuse lexical + vector hit lists (RRF or weighted merge)."""
        s = self.state
        _, pool = rerank_candidates(
            query=s.query,
            fts_hits=s.fts_hits,
            vector_hits=s.vector_hits,
            top_k=s.top_k,
        )
        s.merged_pool = list(pool)

    def stage_scoring(self) -> None:
        """Stage 3: intent-aware structural blend over the merged pool."""
        s = self.state
        s.scored_pool = apply_claude_intent_rerank(s.conn, s.merged_pool, s.query, s.intent)

    def stage_final_selection(self) -> None:
        """Stage 4: take top-k, then apply session delta (omit unchanged redundant chunks)."""
        s = self.state
        pre = s.scored_pool[: s.top_k]
        active_fps = load_active_fingerprints(s.memory_blob, s.session_id)
        s.selected, s.omitted_redundant = partition_redundant_candidates(s.conn, active_fps, pre)

    def stage_expansion(self) -> None:
        """Stage 5: LSP-backed definition expansion for the top surviving candidate."""
        s = self.state
        s.referenced_symbols = []
        if not s.selected:
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
                for name, line, col in function_call_positions(top.text, limit=3):
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
        """Stage 6: knapsack-style compression (TextRank path lives inside compressor)."""
        s = self.state
        if not s.selected and s.omitted_redundant:
            s.bullets = [
                f"(delta) {len(s.omitted_redundant)} chunk(s) status=omitted_redundant: "
                "already in active context (file hash/mtime unchanged). No new compressed text."
            ]
        else:
            s.bullets = compress_candidates(
                query=s.query,
                candidates=s.selected,
                word_budget=s.word_budget,
                relevance_floor=s.relevance_floor,
            )
        s.claude_context = build_claude_plugin_payload(s.query, s.intent, s.selected)

    def run_through_compression(self) -> None:
        """Execute stages 1–6 in order (retrieval … compression)."""
        self.stage_retrieval()
        self.stage_merge()
        self.stage_scoring()
        self.stage_final_selection()
        self.stage_expansion()
        self.stage_compression()


def build_packet_from_state(
    state: ContextRunState,
    *,
    retrieval_mode: str,
    hybrid_mode: str,
    active_sig: str,
) -> ContextPacket:
    """Stage 7: assemble the context packet for the session/UI."""
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
    )
