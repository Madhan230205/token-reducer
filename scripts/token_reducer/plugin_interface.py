"""JSON-first plugin API for Claude tools (local-only)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .chunker import read_text_file
from .config import DEFAULT_DIMENSIONS, DEFAULT_FTS_K
from .db import connect_db
from .embeddings import resolve_embedding_backend
from .pipeline import run_retrieval_pipeline
from .plugin_output import plugin_json_dumps
from .plugin_settings import get_runtime_defaults
from .retriever import fts_retrieve


def get_context(
    conn: sqlite3.Connection,
    db_path: Path,
    query: str,
    *,
    top_k: int | None = None,
    session_id: str = "plugin",
) -> dict[str, object]:
    rt = get_runtime_defaults()
    tk = top_k or rt.default_top_k
    backend, model = resolve_embedding_backend(
        requested_backend="hash",
        requested_model=None,
    )
    pkt = run_retrieval_pipeline(
        conn=conn,
        db_path=db_path,
        query=query,
        top_k=tk,
        fts_k=DEFAULT_FTS_K,
        vector_k=24,
        min_fts_hits=2,
        hybrid_mode="fallback",
        retrieval_mode="compact",
        embedding_backend=backend,
        embedding_model=model,
        session_id=session_id,
        query_cache_ttl_seconds=300,
        dimensions=DEFAULT_DIMENSIONS,
        word_budget=rt.compression_word_budget,
        relevance_floor=rt.relevance_floor,
    )
    ctx = pkt.claude_context or {}
    return {"ok": True, "claude_context": ctx, "metrics": pkt.token_metrics.model_dump()}


def search_code(conn: sqlite3.Connection, query: str, *, top_k: int = 20) -> dict[str, object]:
    hits = fts_retrieve(conn, query, limit=top_k)
    return {
        "ok": True,
        "hits": [
            {
                "file": h.source,
                "chunk_id": h.chunk_id,
                "bm25": h.bm25_score,
                "preview": h.text[:800],
            }
            for h in hits
        ],
    }


def explain_file(file_path: str, *, max_chars: int = 32000) -> dict[str, object]:
    p = Path(file_path).expanduser().resolve()
    raw = read_text_file(p)
    if raw is None:
        return {"ok": False, "error": "unreadable", "path": str(p)}
    body = raw if len(raw) <= max_chars else raw[:max_chars] + "\n…"
    return {"ok": True, "path": str(p), "content": body}


def find_related(conn: sqlite3.Connection, symbol: str, *, top_k: int = 15) -> dict[str, object]:
    q = f'"{symbol}"'
    hits = fts_retrieve(conn, q, limit=top_k)
    return {
        "ok": True,
        "symbol": symbol,
        "occurrences": [{"file": h.source, "preview": h.text[:600]} for h in hits],
    }


def get_context_json(conn: sqlite3.Connection, db_path: Path, query: str, **kw: object) -> str:
    return plugin_json_dumps(get_context(conn, db_path, query, **kw))  # type: ignore[arg-type]


def open_index(db_path: Path) -> sqlite3.Connection:
    return connect_db(db_path)
