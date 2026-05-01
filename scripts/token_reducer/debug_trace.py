"""Debug instrumentation for hybrid retrieval + intent ranking."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .config import should_skip_vector_for_hash
from .intent import analyze_query_intent
from .models import Candidate
from .ranking import apply_claude_intent_rerank
from .retriever import (
    bm25_to_score,
    fts_retrieve,
    infer_retrieval_tier,
    rerank_candidates,
    vector_retrieve,
)


def debug_retrieval_trace(
    conn: sqlite3.Connection,
    db_path: Path,
    query: str,
    *,
    fts_k: int,
    vector_k: int,
    min_fts_hits: int,
    hybrid_mode: str,
    embedding_backend: str,
    embedding_model: str | None,
    dimensions: int,
    top_k: int,
) -> dict[str, object]:
    t0 = time.perf_counter()
    intent = analyze_query_intent(query)
    fts_hits = fts_retrieve(conn, query, limit=fts_k)
    vector_hits: list[Candidate] = []
    vector_path = "disabled"
    adaptive = infer_retrieval_tier(conn)
    use_vector = hybrid_mode == "always" or (
        adaptive != "fts_only" and len(fts_hits) < min_fts_hits
    )
    if use_vector and embedding_backend == "hash" and should_skip_vector_for_hash():
        use_vector = False
        vector_path = "skipped_hash_backend"
    if use_vector:
        vector_hits, _, _, vector_path = vector_retrieve(
            conn=conn,
            db_path=db_path,
            query=query,
            limit=vector_k,
            dimensions=dimensions,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
        )
    _, pool = rerank_candidates(
        query=query, fts_hits=fts_hits, vector_hits=vector_hits, top_k=top_k
    )
    ranked = apply_claude_intent_rerank(conn, list(pool), query, intent)
    ms = (time.perf_counter() - t0) * 1000.0
    rows: list[dict[str, object]] = []
    for c in ranked[:top_k]:
        sem = max(float(c.vector_score), float(c.fts_score) * 0.85)
        rows.append(
            {
                "file": c.source,
                "chunk_id": c.chunk_id,
                "bm25_component": bm25_to_score(c.bm25_score),
                "semantic_component": sem,
                "structural_component": float(c.structural_score),
                "final_score": float(c.final_score),
                "why": f"intent={intent}; fts_rank={c.fts_rank}; vec_rank={c.vector_rank}",
            }
        )
    return {
        "intent": intent,
        "latency_ms": round(ms, 2),
        "adaptive_tier": adaptive,
        "vector_path": vector_path,
        "chunks": rows,
    }
