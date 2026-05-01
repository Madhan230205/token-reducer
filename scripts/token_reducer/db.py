from __future__ import annotations

import contextlib
import json
import sqlite3
import sys
from collections.abc import Iterable
from pathlib import Path

from .chunker import (
    chunk_code,
    chunk_text,
    clean_text,
    estimate_tokens,
    is_code_file,
    is_minified,
    read_text_file,
)
from .embeddings import embed_text
from .models import (
    hash_text,
    utc_now_epoch,
    utc_now_iso,
)
from .structure import meta_json_from_chunk


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL UNIQUE,
                raw_text TEXT NOT NULL,
                cleaned_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                file_mtime REAL,
                file_hash TEXT
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                token_estimate INTEGER NOT NULL,
                meta_json TEXT,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE(document_id, chunk_index)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(document_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(text, content='chunks', content_rowid='id');

            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id INTEGER PRIMARY KEY,
                embedding_json TEXT NOT NULL,
                backend TEXT NOT NULL DEFAULT 'hash',
                dimensions INTEGER NOT NULL DEFAULT 256,
                model_name TEXT,
                FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_backend_dims
            ON chunk_embeddings(backend, dimensions);

            CREATE TABLE IF NOT EXISTS query_embeddings (
                query_key TEXT PRIMARY KEY,
                query_text_hash TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                backend TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                model_name TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_query_embeddings_backend_dims
            ON query_embeddings(backend, dimensions);

            CREATE TABLE IF NOT EXISTS query_cache (
                cache_key TEXT PRIMARY KEY,
                cache_payload_json TEXT NOT NULL,
                created_at_epoch INTEGER NOT NULL,
                expires_at_epoch INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_query_cache_expires
            ON query_cache(expires_at_epoch);
            """
        )

        for statement in (
            "ALTER TABLE chunk_embeddings ADD COLUMN backend TEXT NOT NULL DEFAULT 'hash'",
            "ALTER TABLE chunk_embeddings ADD COLUMN dimensions INTEGER NOT NULL DEFAULT 256",
            "ALTER TABLE chunk_embeddings ADD COLUMN model_name TEXT",
            "ALTER TABLE documents ADD COLUMN file_mtime REAL",
            "ALTER TABLE documents ADD COLUMN file_hash TEXT",
            "ALTER TABLE chunks ADD COLUMN meta_json TEXT",
        ):
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(statement)

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_backend_dims ON chunk_embeddings(backend, dimensions);"
        )
    except sqlite3.OperationalError as exc:
        if "fts5" in str(exc).lower():
            raise RuntimeError("SQLite FTS5 is not available in this Python build.") from exc
        raise

    return conn


def get_index_fingerprint(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        """
        SELECT COUNT(*) AS chunk_count, COALESCE(MAX(updated_at), '') AS max_updated
        FROM documents
        """
    ).fetchone()
    chunk_row = conn.execute("SELECT COUNT(*) AS total FROM chunks").fetchone()
    chunk_count = int(chunk_row["total"]) if chunk_row else 0
    doc_count = int(row["chunk_count"]) if row and row["chunk_count"] is not None else 0
    max_updated = str(row["max_updated"]) if row and row["max_updated"] is not None else ""
    return hash_text(f"docs={doc_count}|chunks={chunk_count}|updated={max_updated}")


def cleanup_query_cache(conn: sqlite3.Connection, now_epoch: int) -> None:
    conn.execute("DELETE FROM query_cache WHERE expires_at_epoch <= ?", (now_epoch,))
    conn.commit()


def get_cached_query_result(
    conn: sqlite3.Connection, cache_key: str, now_epoch: int
) -> dict | None:
    row = conn.execute(
        """
        SELECT cache_payload_json, expires_at_epoch
        FROM query_cache
        WHERE cache_key = ?
        """,
        (cache_key,),
    ).fetchone()
    if not row:
        return None

    if int(row["expires_at_epoch"]) <= now_epoch:
        conn.execute("DELETE FROM query_cache WHERE cache_key = ?", (cache_key,))
        conn.commit()
        return None

    try:
        return json.loads(str(row["cache_payload_json"]))
    except Exception:
        return None


def set_cached_query_result(
    conn: sqlite3.Connection,
    cache_key: str,
    payload: dict,
    now_epoch: int,
    ttl_seconds: int,
) -> None:
    expires_at = now_epoch + max(30, ttl_seconds)
    conn.execute(
        """
        INSERT INTO query_cache (cache_key, cache_payload_json, created_at_epoch, expires_at_epoch)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
            cache_payload_json = excluded.cache_payload_json,
            created_at_epoch = excluded.created_at_epoch,
            expires_at_epoch = excluded.expires_at_epoch
        """,
        (cache_key, json.dumps(payload, separators=(",", ":")), now_epoch, expires_at),
    )
    conn.commit()


def get_cached_query_embedding(conn: sqlite3.Connection, query_key: str) -> list[float] | None:
    row = conn.execute(
        "SELECT embedding_json FROM query_embeddings WHERE query_key = ?",
        (query_key,),
    ).fetchone()
    if not row:
        return None
    try:
        return [float(x) for x in json.loads(str(row["embedding_json"]))]
    except Exception:
        return None


def set_cached_query_embedding(
    conn: sqlite3.Connection,
    query_key: str,
    query_text_hash: str,
    embedding: list[float],
    backend: str,
    model_name: str | None,
) -> None:
    now_iso = utc_now_iso()
    conn.execute(
        """
        INSERT INTO query_embeddings (
            query_key,
            query_text_hash,
            embedding_json,
            backend,
            dimensions,
            model_name,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(query_key) DO UPDATE SET
            embedding_json = excluded.embedding_json,
            backend = excluded.backend,
            dimensions = excluded.dimensions,
            model_name = excluded.model_name,
            updated_at = excluded.updated_at
        """,
        (
            query_key,
            query_text_hash,
            json.dumps(embedding, separators=(",", ":")),
            backend,
            len(embedding),
            model_name,
            now_iso,
            now_iso,
        ),
    )
    conn.commit()


def session_memory_path(db_path: Path) -> Path:
    return db_path.parent / "session_memory.json"


def load_session_memory(path: Path) -> dict:
    if not path.exists():
        return {"sessions": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"sessions": {}}


def save_session_memory(path: Path, memory: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory, ensure_ascii=False), encoding="utf-8")


def get_recent_session_queries(memory: dict, session_id: str, limit: int = 4) -> list[str]:
    sessions = memory.get("sessions", {})
    session = sessions.get(session_id, {})
    recent = session.get("recent_queries", [])
    if not isinstance(recent, list):
        return []
    return [str(x) for x in recent[-limit:]]


def update_session_memory(
    memory_path: Path,
    session_id: str,
    query: str,
    selected_sources: list[str],
) -> dict:
    memory = load_session_memory(memory_path)
    sessions = memory.setdefault("sessions", {})
    session = sessions.setdefault(session_id, {"recent_queries": [], "recent_sources": []})

    recent_queries = session.setdefault("recent_queries", [])
    recent_queries.append(query)
    session["recent_queries"] = recent_queries[-8:]

    recent_sources = session.setdefault("recent_sources", [])
    for src in selected_sources:
        if src not in recent_sources:
            recent_sources.append(src)
    session["recent_sources"] = recent_sources[-12:]

    session["updated_at"] = utc_now_iso()
    sessions[session_id] = session
    memory["sessions"] = sessions
    save_session_memory(memory_path, memory)
    return memory


def upsert_document(
    conn: sqlite3.Connection,
    source: str,
    raw_text: str,
    cleaned_text: str,
    chunk_size_words: int,
    overlap_words: int,
    dimensions: int,
    embedding_backend: str,
    embedding_model: str | None,
) -> int:
    now = utc_now_iso()
    file_hash = hash_text(raw_text)
    try:
        src_path = Path(source)
        file_mtime = src_path.stat().st_mtime if src_path.is_file() else None
    except OSError:
        file_mtime = None

    existing = conn.execute("SELECT id FROM documents WHERE source = ?", (source,)).fetchone()

    if existing:
        document_id = int(existing["id"])
        old_chunks = conn.execute(
            "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
        ).fetchall()
        for row in old_chunks:
            cid = int(row["id"])
            conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (cid,))
            conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

        conn.execute(
            """
            UPDATE documents
            SET raw_text = ?, cleaned_text = ?, updated_at = ?, file_hash = ?, file_mtime = ?
            WHERE id = ?
            """,
            (raw_text, cleaned_text, now, file_hash, file_mtime, document_id),
        )
    else:
        cursor = conn.execute(
            """
            INSERT INTO documents (source, raw_text, cleaned_text, created_at, updated_at, file_hash, file_mtime)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (source, raw_text, cleaned_text, now, now, file_hash, file_mtime),
        )
        document_id = int(cursor.lastrowid)

    if is_code_file(source):
        chunks = chunk_code(cleaned_text, source, chunk_size_words)
    else:
        chunks = chunk_text(cleaned_text, chunk_size_words, overlap_words)
    for idx, chunk in enumerate(chunks):
        meta = meta_json_from_chunk(chunk, source)
        cursor = conn.execute(
            """
            INSERT INTO chunks (document_id, chunk_index, text, token_estimate, meta_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (document_id, idx, chunk, estimate_tokens(chunk), meta),
        )
        chunk_id = int(cursor.lastrowid)
        conn.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", (chunk_id, chunk))
        emb, effective_backend, effective_model = embed_text(
            text=chunk,
            dimensions=dimensions,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
        )
        conn.execute(
            """
            INSERT INTO chunk_embeddings (chunk_id, embedding_json, backend, dimensions, model_name)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                json.dumps(emb, separators=(",", ":")),
                effective_backend,
                len(emb),
                effective_model,
            ),
        )

    return len(chunks)


def index_corpus(
    conn: sqlite3.Connection,
    file_paths: Iterable[Path],
    chunk_size_words: int,
    overlap_words: int,
    dimensions: int,
    embedding_backend: str,
    embedding_model: str | None,
) -> dict[str, int]:
    files_indexed = 0
    chunks_indexed = 0

    for path in file_paths:
        raw = read_text_file(path)
        if raw is None:
            print(f"[warn] Could not read file: {path}", file=sys.stderr)
            continue

        if is_minified(raw):
            print(f"[skip] Minified content detected: {path.name}", file=sys.stderr)
            continue

        cleaned = clean_text(raw)
        if not cleaned:
            continue

        chunk_count = upsert_document(
            conn=conn,
            source=str(path),
            raw_text=raw,
            cleaned_text=cleaned,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            dimensions=dimensions,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
        )
        files_indexed += 1
        chunks_indexed += chunk_count

    conn.commit()
    return {"files_indexed": files_indexed, "chunks_indexed": chunks_indexed}


# ---------------------------------------------------------------------------
# Context Lifecycle Management Functions
# ---------------------------------------------------------------------------


def get_index_stats(conn: sqlite3.Connection) -> dict:
    """Get comprehensive statistics about the index."""
    doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    embedding_count = conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
    cache_count = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
    query_emb_count = conn.execute("SELECT COUNT(*) FROM query_embeddings").fetchone()[0]

    # Get oldest and newest document timestamps
    date_row = conn.execute("SELECT MIN(created_at), MAX(updated_at) FROM documents").fetchone()
    oldest_doc = date_row[0] if date_row else None
    newest_doc = date_row[1] if date_row else None

    # Get embedding backend distribution
    backend_rows = conn.execute(
        "SELECT backend, COUNT(*) FROM chunk_embeddings GROUP BY backend"
    ).fetchall()
    backends = {row[0]: row[1] for row in backend_rows}

    return {
        "documents": doc_count,
        "chunks": chunk_count,
        "embeddings": embedding_count,
        "query_cache_entries": cache_count,
        "query_embeddings": query_emb_count,
        "oldest_document": oldest_doc,
        "newest_document": newest_doc,
        "embedding_backends": backends,
    }


def get_indexed_files(conn: sqlite3.Connection) -> dict[str, tuple[float | None, str | None]]:
    """Get all indexed files with their mtime and hash."""
    rows = conn.execute("SELECT source, file_mtime, file_hash FROM documents").fetchall()
    return {row[0]: (row[1], row[2]) for row in rows}


def detect_file_changes(
    conn: sqlite3.Connection,
    file_paths: list[Path],
) -> tuple[list[Path], list[Path], list[str]]:
    """Detect new, modified, and deleted files compared to index.

    Returns:
        Tuple of (new_files, modified_files, deleted_sources)
    """
    indexed = get_indexed_files(conn)
    indexed_sources = set(indexed.keys())

    new_files: list[Path] = []
    modified_files: list[Path] = []

    for path in file_paths:
        source = str(path)
        if source not in indexed_sources:
            new_files.append(path)
        else:
            stored_mtime, stored_hash = indexed[source]
            try:
                current_mtime = path.stat().st_mtime
                if stored_mtime is None or current_mtime > stored_mtime:
                    modified_files.append(path)
            except OSError:
                # File may have been deleted or is inaccessible
                continue

    # Find deleted files (in index but not in provided file list)
    current_sources = {str(p) for p in file_paths}
    deleted_sources = [s for s in indexed_sources if s not in current_sources]

    return new_files, modified_files, deleted_sources


def remove_documents_by_source(conn: sqlite3.Connection, sources: list[str]) -> int:
    """Remove documents and their associated data by source path."""
    if not sources:
        return 0

    removed = 0
    for source in sources:
        doc_row = conn.execute("SELECT id FROM documents WHERE source = ?", (source,)).fetchone()
        if doc_row:
            doc_id = doc_row[0]
            # Get chunk IDs for cleanup
            chunk_rows = conn.execute(
                "SELECT id FROM chunks WHERE document_id = ?", (doc_id,)
            ).fetchall()
            for chunk_row in chunk_rows:
                cid = chunk_row[0]
                conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (cid,))
                conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            removed += 1

    conn.commit()
    return removed


def garbage_collect(
    conn: sqlite3.Connection,
    max_cache_age_seconds: int = 86400,  # 24 hours default
    dry_run: bool = False,
) -> dict:
    """Perform garbage collection on the database.

    Cleans up:
    - Orphaned chunks (no parent document)
    - Orphaned embeddings (no parent chunk)
    - Expired query cache entries
    - Stale session memory entries

    Returns statistics about what was (or would be) cleaned.
    """
    stats = {
        "orphaned_chunks": 0,
        "orphaned_embeddings": 0,
        "expired_cache_entries": 0,
        "stale_query_embeddings": 0,
    }

    # Find orphaned chunks (chunks without documents)
    orphan_chunks = conn.execute(
        """
        SELECT COUNT(*) FROM chunks c
        WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = c.document_id)
        """
    ).fetchone()[0]
    stats["orphaned_chunks"] = orphan_chunks

    # Find orphaned embeddings (embeddings without chunks)
    orphan_embeddings = conn.execute(
        """
        SELECT COUNT(*) FROM chunk_embeddings ce
        WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.id = ce.chunk_id)
        """
    ).fetchone()[0]
    stats["orphaned_embeddings"] = orphan_embeddings

    # Find expired cache entries
    now_epoch = utc_now_epoch()
    expired_cache = conn.execute(
        "SELECT COUNT(*) FROM query_cache WHERE expires_at_epoch <= ?", (now_epoch,)
    ).fetchone()[0]
    stats["expired_cache_entries"] = expired_cache

    if not dry_run:
        # Delete orphaned chunks
        conn.execute(
            """
            DELETE FROM chunks WHERE id IN (
                SELECT c.id FROM chunks c
                WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = c.document_id)
            )
            """
        )

        # Delete orphaned embeddings
        conn.execute(
            """
            DELETE FROM chunk_embeddings WHERE chunk_id IN (
                SELECT ce.chunk_id FROM chunk_embeddings ce
                WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.id = ce.chunk_id)
            )
            """
        )

        # Delete expired cache entries
        conn.execute("DELETE FROM query_cache WHERE expires_at_epoch <= ?", (now_epoch,))

        conn.commit()

    return stats


def vacuum_database(conn: sqlite3.Connection) -> None:
    """Vacuum the database to reclaim space."""
    conn.execute("VACUUM")


def get_database_size(db_path: Path) -> int:
    """Get the database file size in bytes."""
    if db_path.exists() and db_path.name != ":memory:":
        return db_path.stat().st_size
    return 0


def update_document_mtime(
    conn: sqlite3.Connection,
    source: str,
    mtime: float,
    file_hash: str | None = None,
) -> None:
    """Update the mtime and hash for a document."""
    if file_hash:
        conn.execute(
            "UPDATE documents SET file_mtime = ?, file_hash = ? WHERE source = ?",
            (mtime, file_hash, source),
        )
    else:
        conn.execute(
            "UPDATE documents SET file_mtime = ? WHERE source = ?",
            (mtime, source),
        )
