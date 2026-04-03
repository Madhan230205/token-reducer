from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from .config import (
    ADAPTIVE_TIER_SMALL_CHUNKS,
    ADAPTIVE_TIER_MEDIUM_CHUNKS,
    _EMBEDDING_VECTOR_CACHE,
    get_weight,
    should_skip_vector_for_hash,
)
from .models import Candidate, embedding_cache_key, hash_text
from .embeddings import embed_text, cosine_similarity
from .db import (
    get_cached_query_embedding,
    set_cached_query_embedding,
)
from .ann import query_hnsw_index, query_faiss_index


def infer_retrieval_tier(conn: sqlite3.Connection) -> str:
    """Dynamically determine the retrieval tier from indexed chunk count.

    Returns one of:
    - "fts_only"     → small codebase  (< ADAPTIVE_TIER_SMALL_CHUNKS chunks)
    - "fts_with_hash"→ medium codebase (< ADAPTIVE_TIER_MEDIUM_CHUNKS chunks)
    - "full_hybrid"  → large codebase  (>= ADAPTIVE_TIER_MEDIUM_CHUNKS chunks)

    This drives whether ANN is built and whether vector retrieval is used.
    """
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        count = int(row["n"]) if row else 0
    except Exception:
        count = 0
    if count < ADAPTIVE_TIER_SMALL_CHUNKS:
        return "fts_only"
    if count < ADAPTIVE_TIER_MEDIUM_CHUNKS:
        return "fts_with_hash"
    return "full_hybrid"


def build_fts_query(query: str) -> str:
    from .chunker import tokenize
    terms = tokenize(query)[:20]
    if not terms:
        return query.strip()
    return " OR ".join(f'"{term}"' for term in terms)


def fts_retrieve(conn: sqlite3.Connection, query: str, limit: int) -> list[Candidate]:
    fts_query = build_fts_query(query)
    if not fts_query:
        return []

    try:
        rows = conn.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.text AS text,
                c.chunk_index AS chunk_index,
                c.token_estimate AS token_estimate,
                d.source AS source,
                bm25(chunks_fts) AS bm25_score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.id = c.document_id
            WHERE chunks_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        from .chunker import tokenize
        safe = " ".join(tokenize(query)[:20])
        if not safe:
            return []
        rows = conn.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.text AS text,
                c.chunk_index AS chunk_index,
                c.token_estimate AS token_estimate,
                d.source AS source,
                bm25(chunks_fts) AS bm25_score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.id = c.document_id
            WHERE chunks_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
            """,
            (safe, limit),
        ).fetchall()

    results: list[Candidate] = []
    for i, row in enumerate(rows, start=1):
        results.append(
            Candidate(
                chunk_id=int(row["chunk_id"]),
                source=str(row["source"]),
                chunk_index=int(row["chunk_index"]),
                text=str(row["text"]),
                token_estimate=int(row["token_estimate"]),
                bm25_score=float(row["bm25_score"]) if row["bm25_score"] is not None else None,
                fts_rank=i,
            )
        )
    return results


def vector_retrieve(
    conn: sqlite3.Connection,
    db_path: Path,
    query: str,
    limit: int,
    dimensions: int,
    embedding_backend: str,
    embedding_model: str | None,
) -> tuple[list[Candidate], str, str | None, str]:
    q_vec, effective_backend, effective_model = get_query_embedding(
        conn=conn,
        query=query,
        dimensions=dimensions,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
    )
    effective_dimensions = len(q_vec)

    def fetch_rows(backend: str, vector_dims: int, model_name: str | None):
        base_sql = """
        SELECT
            c.id AS chunk_id,
            c.text AS text,
            c.chunk_index AS chunk_index,
            c.token_estimate AS token_estimate,
            d.source AS source,
            ce.embedding_json AS embedding_json,
            ce.backend AS backend,
            ce.dimensions AS dimensions,
            ce.model_name AS model_name
        FROM chunk_embeddings ce
        JOIN chunks c ON c.id = ce.chunk_id
        JOIN documents d ON d.id = c.document_id
        WHERE ce.backend = ? AND ce.dimensions = ?
        """
        params: list[object] = [backend, vector_dims]
        if backend == "ml" and model_name:
            base_sql += " AND (ce.model_name = ? OR ce.model_name IS NULL)"
            params.append(model_name)
        return conn.execute(base_sql, tuple(params)).fetchall()

    ann_results = query_hnsw_index(
        conn=conn,
        db_path=db_path,
        query_vec=q_vec,
        limit=limit,
        backend=effective_backend,
        model_name=effective_model,
    )

    if ann_results:
        chunk_ids = [cid for cid, _ in ann_results]
        placeholder = ",".join("?" for _ in chunk_ids)
        detail_rows = conn.execute(
            f"""
            SELECT
                c.id AS chunk_id,
                c.text AS text,
                c.chunk_index AS chunk_index,
                c.token_estimate AS token_estimate,
                d.source AS source
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholder})
            """,
            tuple(chunk_ids),
        ).fetchall()
        detail_map = {int(r["chunk_id"]): r for r in detail_rows}

        top: list[Candidate] = []
        for i, (chunk_id, sim) in enumerate(ann_results, start=1):
            row = detail_map.get(chunk_id)
            if row is None:
                continue
            top.append(
                Candidate(
                    chunk_id=int(row["chunk_id"]),
                    source=str(row["source"]),
                    chunk_index=int(row["chunk_index"]),
                    text=str(row["text"]),
                    token_estimate=int(row["token_estimate"]),
                    vector_score=float(sim),
                    vector_rank=i,
                )
            )
        if top:
            return top, effective_backend, effective_model, "hnsw"

    # Try FAISS as second ANN engine if HNSW didn't produce results
    faiss_results = query_faiss_index(
        conn=conn,
        db_path=db_path,
        query_vec=q_vec,
        limit=limit,
        backend=effective_backend,
        model_name=effective_model,
    )
    if faiss_results:
        chunk_ids = [cid for cid, _ in faiss_results]
        placeholder = ",".join("?" for _ in chunk_ids)
        detail_rows = conn.execute(
            f"""
            SELECT
                c.id AS chunk_id,
                c.text AS text,
                c.chunk_index AS chunk_index,
                c.token_estimate AS token_estimate,
                d.source AS source
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholder})
            """,
            tuple(chunk_ids),
        ).fetchall()
        detail_map = {int(r["chunk_id"]): r for r in detail_rows}
        top: list[Candidate] = []
        for i, (chunk_id, sim) in enumerate(faiss_results, start=1):
            row = detail_map.get(chunk_id)
            if row is None:
                continue
            top.append(
                Candidate(
                    chunk_id=int(row["chunk_id"]),
                    source=str(row["source"]),
                    chunk_index=int(row["chunk_index"]),
                    text=str(row["text"]),
                    token_estimate=int(row["token_estimate"]),
                    vector_score=float(sim),
                    vector_rank=i,
                )
            )
        if top:
            return top, effective_backend, effective_model, "faiss"

    rows = fetch_rows(effective_backend, effective_dimensions, effective_model)

    if not rows and effective_backend == "ml":
        print(
            "[warn] No ML embeddings found in index for selected model; falling back to hash vector retrieval.",
            file=sys.stderr,
        )
        q_vec, effective_backend, effective_model = get_query_embedding(
            conn=conn,
            query=query,
            dimensions=dimensions,
            embedding_backend="hash",
            embedding_model=None,
        )
        effective_dimensions = len(q_vec)
        rows = fetch_rows(effective_backend, effective_dimensions, effective_model)

    scored: list[tuple[float, Candidate]] = []
    for row in rows:
        try:
            emb = json.loads(str(row["embedding_json"]))
        except json.JSONDecodeError:
            continue

        sim = cosine_similarity(q_vec, emb)
        if sim <= 0:
            continue

        scored.append(
            (
                sim,
                Candidate(
                    chunk_id=int(row["chunk_id"]),
                    source=str(row["source"]),
                    chunk_index=int(row["chunk_index"]),
                    text=str(row["text"]),
                    token_estimate=int(row["token_estimate"]),
                    vector_score=sim,
                ),
            )
        )

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = [candidate for _, candidate in scored[:limit]]
    for i, candidate in enumerate(top, start=1):
        candidate.vector_rank = i
    return top, effective_backend, effective_model, "scan"


def rank_score(rank: int | None) -> float:
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / float(rank)


def bm25_to_score(bm25_value: float | None) -> float:
    if bm25_value is None:
        return 0.0
    return 1.0 / (1.0 + abs(float(bm25_value)))


def overlap_ratio(query: str, text: str) -> float:
    from .chunker import tokenize
    q_terms = set(tokenize(query))
    if not q_terms:
        return 0.0
    t_terms = set(tokenize(text))
    return len(q_terms & t_terms) / float(len(q_terms))


def rerank_candidates(
    query: str,
    fts_hits: list[Candidate],
    vector_hits: list[Candidate],
    top_k: int,
) -> tuple[list[Candidate], list[Candidate]]:
    merged: dict[int, Candidate] = {}

    for candidate in fts_hits:
        merged[candidate.chunk_id] = candidate

    for candidate in vector_hits:
        existing = merged.get(candidate.chunk_id)
        if existing is None:
            merged[candidate.chunk_id] = candidate
        else:
            existing.vector_rank = candidate.vector_rank
            existing.vector_score = candidate.vector_score

    # Get configurable weights
    fts_lexical_w = get_weight("fts_lexical_rank_weight")
    fts_bm25_w = get_weight("fts_bm25_weight")
    final_fts_w = get_weight("final_fts_weight")
    final_vector_w = get_weight("final_vector_weight")
    final_overlap_w = get_weight("final_overlap_weight")

    ranked = list(merged.values())
    for item in ranked:
        lexical_rank_signal = rank_score(item.fts_rank)
        bm25_signal = bm25_to_score(item.bm25_score)
        item.fts_score = (fts_lexical_w * lexical_rank_signal) + (fts_bm25_w * bm25_signal)
        item.vector_score = item.vector_score if item.vector_score > 0 else rank_score(item.vector_rank)
        item.overlap_score = overlap_ratio(query, item.text)
        item.final_score = (final_fts_w * item.fts_score) + (final_vector_w * item.vector_score) + (final_overlap_w * item.overlap_score)

    ranked.sort(key=lambda c: c.final_score, reverse=True)
    # Allow full top_k pool to pass to the compressor;
    # relevance floor in compressor will filter low-quality chunks
    return ranked[:top_k], ranked


def get_query_embedding(
    conn: sqlite3.Connection,
    query: str,
    dimensions: int,
    embedding_backend: str,
    embedding_model: str | None,
) -> tuple[list[float], str, str | None]:
    requested_key = embedding_cache_key(
        text=query,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        dimensions=dimensions,
    )
    in_memory = _EMBEDDING_VECTOR_CACHE.get(requested_key)
    if in_memory is not None:
        return in_memory, embedding_backend, embedding_model

    persisted = get_cached_query_embedding(conn=conn, query_key=requested_key)
    if persisted is not None:
        _EMBEDDING_VECTOR_CACHE[requested_key] = persisted
        return persisted, embedding_backend, embedding_model

    vec, effective_backend, effective_model = embed_text(
        text=query,
        dimensions=dimensions,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
    )

    effective_key = embedding_cache_key(
        text=query,
        embedding_backend=effective_backend,
        embedding_model=effective_model,
        dimensions=len(vec),
    )
    query_hash = hash_text(query)

    _EMBEDDING_VECTOR_CACHE[effective_key] = vec
    if effective_key != requested_key:
        _EMBEDDING_VECTOR_CACHE[requested_key] = vec

    set_cached_query_embedding(
        conn=conn,
        query_key=effective_key,
        query_text_hash=query_hash,
        embedding=vec,
        backend=effective_backend,
        model_name=effective_model,
    )
    if effective_key != requested_key:
        set_cached_query_embedding(
            conn=conn,
            query_key=requested_key,
            query_text_hash=query_hash,
            embedding=vec,
            backend=effective_backend,
            model_name=effective_model,
        )

    return vec, effective_backend, effective_model
