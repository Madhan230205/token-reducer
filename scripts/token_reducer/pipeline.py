from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .config import DEFAULT_RELEVANCE_FLOOR, MAX_QUERY_LINES, MAX_QUERY_WORDS
from .db import (
    cleanup_query_cache,
    get_cached_query_result,
    get_index_fingerprint,
    get_recent_session_queries,
    load_session_memory,
    session_memory_path,
    set_cached_query_result,
    update_session_memory,
)
from .delta_context import (
    active_context_count,
    active_context_signature,
    fingerprint_dict_for_chunk,
    persist_delivered_fingerprints,
)
from .intent import analyze_query_intent
from .models import CacheInfo, ContextPacket, SessionMemory, hash_text, utc_now_epoch
from .orchestrator import (
    ContextPipelineOrchestrator,
    ContextRunState,
    build_packet_from_state,
)
from .plugin_settings import get_runtime_defaults


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
    workspace_root: Path | None = None,
) -> ContextPacket:
    runtime = get_runtime_defaults()
    intent = analyze_query_intent(query)
    now_epoch = utc_now_epoch()
    cleanup_query_cache(conn=conn, now_epoch=now_epoch)

    memory_path = session_memory_path(db_path)
    memory_blob = load_session_memory(memory_path)
    active_sig = active_context_signature(memory_blob, session_id)

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
            "relevance_floor": relevance_floor,
            "session_id": session_id,
            "index_fingerprint": index_fingerprint,
            "active_context_sig": active_sig,
            "lsp_servers_sig": hash_text(
                json.dumps(
                    {k: list(v) for k, v in sorted(runtime.lsp_servers.items())},
                    sort_keys=True,
                )
            ),
            "intent": intent,
        },
        sort_keys=True,
    )
    query_cache_key = hash_text(cache_key_material)
    cached_result = get_cached_query_result(
        conn=conn, cache_key=query_cache_key, now_epoch=now_epoch
    )
    if cached_result is not None:
        updated_memory = update_session_memory(
            memory_path=memory_path,
            session_id=session_id,
            query=query,
            selected_sources=[
                str(item.get("source"))
                for item in cached_result.get("candidates", [])
                if isinstance(item, dict)
            ],
        )
        cached_packet = ContextPacket.model_validate(cached_result)
        cached_packet.session_memory = SessionMemory(
            session_id=session_id,
            recent_queries=get_recent_session_queries(
                updated_memory, session_id=session_id, limit=4
            ),
            active_context_tracked=active_context_count(updated_memory, session_id),
        )
        if cached_packet.cache is None:
            cached_packet.cache = CacheInfo()
        cached_packet.cache.hit = True
        cached_packet.cache.key = query_cache_key
        if cached_packet.cache.ttl_seconds == 0:
            cached_packet.cache.ttl_seconds = query_cache_ttl_seconds
        return cached_packet

    # Cache miss: explicit orchestration (retrieval → merge → scoring → selection → expansion → compression → packaging).
    state = ContextRunState(
        conn=conn,
        db_path=db_path,
        query=query,
        intent=intent,
        runtime=runtime,
        memory_blob=memory_blob,
        session_id=session_id,
        top_k=top_k,
        fts_k=fts_k,
        vector_k=vector_k,
        min_fts_hits=min_fts_hits,
        hybrid_mode=hybrid_mode,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        dimensions=dimensions,
        word_budget=word_budget,
        relevance_floor=relevance_floor,
        workspace_root=workspace_root,
    )
    ContextPipelineOrchestrator(state).run_through_compression()
    packet = build_packet_from_state(
        state,
        retrieval_mode=retrieval_mode,
        hybrid_mode=hybrid_mode,
        active_sig=active_sig,
    )

    updated_memory = update_session_memory(
        memory_path=memory_path,
        session_id=session_id,
        query=query,
        selected_sources=[c.source for c in state.selected],
    )
    delivered_fps: list[dict] = []
    for c in state.selected:
        fd = fingerprint_dict_for_chunk(conn, c)
        if fd is not None:
            delivered_fps.append(fd)
    if delivered_fps:
        updated_memory = persist_delivered_fingerprints(memory_path, session_id, delivered_fps)

    packet.session_memory = SessionMemory(
        session_id=session_id,
        recent_queries=get_recent_session_queries(updated_memory, session_id=session_id, limit=4),
        active_context_tracked=active_context_count(updated_memory, session_id),
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
