from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .config import MAX_QUERY_WORDS, MAX_QUERY_LINES, DEFAULT_RELEVANCE_FLOOR
from .models import utc_now_epoch, hash_text, ContextPacket, SessionMemory, CacheInfo
from .db import (
    cleanup_query_cache,
    get_index_fingerprint,
    get_cached_query_result,
    set_cached_query_result,
    session_memory_path,
    update_session_memory,
    get_recent_session_queries,
)
from .ann import build_hnsw_index, infer_embedding_dimensions
from .retriever import (
    infer_retrieval_tier,
    fts_retrieve,
    vector_retrieve,
    rerank_candidates,
)
from .compressor import compress_candidates, build_packet


def validate_query_input(query: str) -> tuple[bool, str | None]:
    word_count = len(query.split())
    line_count = query.count("\n") + 1

    if word_count > MAX_QUERY_WORDS or line_count > MAX_QUERY_LINES:
        return (
            False,
            (
                f"Query looks like pasted raw corpus ({word_count} words, {line_count} lines). "
                "Keep query concise and pass large files/logs via --inputs, then rerun."
            ),
        )

    return True, None


def run_retrieval_pipeline(
    conn: sqlite3.Connection,
    db_path: Path,
    query: str,
    top_k: int,
    fts_k: int,
    vector_k: int,
    min_fts_hits: int,
    hybrid_mode: str,
    retrieval_mode: str,
    embedding_backend: str,
    embedding_model: str | None,
    session_id: str,
    query_cache_ttl_seconds: int,
    dimensions: int,
    word_budget: int,
    relevance_floor: float = DEFAULT_RELEVANCE_FLOOR,
) -> ContextPacket:
    from .config import should_skip_vector_for_hash

    now_epoch = utc_now_epoch()
    cleanup_query_cache(conn=conn, now_epoch=now_epoch)

    index_fingerprint = get_index_fingerprint(conn)
    cache_key_material = json.dumps(
        {
            "query": query,
            "top_k": top_k,
            "fts_k": fts_k,
            "vector_k": vector_k,
            "min_fts_hits": min_fts_hits,
            "hybrid_mode": hybrid_mode,
            "retrieval_mode": retrieval_mode,
            "embedding_backend": embedding_backend,
            "embedding_model": embedding_model,
            "dimensions": dimensions,
            "word_budget": word_budget,
            "session_id": session_id,
            "index_fingerprint": index_fingerprint,
        },
        sort_keys=True,
    )
    query_cache_key = hash_text(cache_key_material)
    cached_result = get_cached_query_result(conn=conn, cache_key=query_cache_key, now_epoch=now_epoch)
    if cached_result is not None:
        memory_path = session_memory_path(db_path)
        updated_memory = update_session_memory(
            memory_path=memory_path,
            session_id=session_id,
            query=query,
            selected_sources=[str(item.get("source")) for item in cached_result.get("candidates", []) if isinstance(item, dict)],
        )
        # Reconstruct ContextPacket from cached dict
        cached_packet = ContextPacket.model_validate(cached_result)
        cached_packet.session_memory = SessionMemory(
            session_id=session_id,
            recent_queries=get_recent_session_queries(updated_memory, session_id=session_id, limit=4),
        )
        if cached_packet.cache is None:
            cached_packet.cache = CacheInfo()
        cached_packet.cache.hit = True
        cached_packet.cache.key = query_cache_key
        if cached_packet.cache.ttl_seconds == 0:
            cached_packet.cache.ttl_seconds = query_cache_ttl_seconds
        return cached_packet

    fts_hits = fts_retrieve(conn, query, limit=fts_k)
    vector_hits = []
    vector_backend_used = "disabled"
    vector_model_used: str | None = None
    vector_retrieval_path = "disabled"

    # Adaptive tier: derive retrieval strategy from codebase size.
    # "always" mode bypasses adaptive logic (explicit user override).
    adaptive_tier = infer_retrieval_tier(conn)
    if hybrid_mode == "always":
        use_vector = True
    elif adaptive_tier == "fts_only":
        # Small codebase — FTS5 alone is sufficient; skip all vector work.
        use_vector = False
        vector_retrieval_path = "disabled_small_codebase"
    elif adaptive_tier == "fts_with_hash":
        # Medium codebase — use hash-embedding fallback only when FTS hits are sparse.
        use_vector = len(fts_hits) < min_fts_hits
    else:
        # Large codebase — standard fallback logic (FTS first, vector when needed).
        use_vector = len(fts_hits) < min_fts_hits

    # Skip vector retrieval when using hash embeddings (redundant with FTS5/BM25)
    if use_vector and embedding_backend == "hash" and should_skip_vector_for_hash():
        use_vector = False
        vector_retrieval_path = "skipped_hash_backend"

    if use_vector:
        vector_hits, vector_backend_used, vector_model_used, vector_retrieval_path = vector_retrieve(
            conn=conn,
            db_path=db_path,
            query=query,
            limit=vector_k,
            dimensions=dimensions,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
        )

    selected, candidate_pool = rerank_candidates(
        query=query,
        fts_hits=fts_hits,
        vector_hits=vector_hits,
        top_k=top_k,
    )
    bullets = compress_candidates(
        query=query,
        candidates=selected,
        word_budget=word_budget,
        relevance_floor=relevance_floor,
    )

    packet = build_packet(
        query=query,
        selected=selected,
        candidate_pool=candidate_pool,
        bullets=bullets,
        fts_hit_count=len(fts_hits),
        vector_hit_count=len(vector_hits),
        hybrid_mode=hybrid_mode,
        retrieval_mode=retrieval_mode,
        vector_backend_used=vector_backend_used,
        vector_model_used=vector_model_used,
        vector_retrieval_path=vector_retrieval_path,
    )

    memory_path = session_memory_path(db_path)
    updated_memory = update_session_memory(
        memory_path=memory_path,
        session_id=session_id,
        query=query,
        selected_sources=[c.source for c in selected],
    )
    packet.session_memory = SessionMemory(
        session_id=session_id,
        recent_queries=get_recent_session_queries(updated_memory, session_id=session_id, limit=4),
    )
    packet.cache = CacheInfo(
        hit=False,
        key=query_cache_key,
        ttl_seconds=query_cache_ttl_seconds,
    )

    set_cached_query_result(
        conn=conn,
        cache_key=query_cache_key,
        payload=packet.model_dump(),
        now_epoch=now_epoch,
        ttl_seconds=query_cache_ttl_seconds,
    )

    return packet
