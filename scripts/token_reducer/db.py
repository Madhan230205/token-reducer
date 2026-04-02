from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

from .models import (
    Candidate,
    utc_now_iso,
    utc_now_epoch,
    hash_text,
    embedding_cache_key,
)
from .chunker import (
    clean_text,
    chunk_text,
    chunk_code,
    is_code_file,
    is_minified,
    read_text_file,
    estimate_tokens,
    extract_imports,
    extract_function_calls,
    resolve_import_to_file,
)
from .embeddings import embed_text


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
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                token_estimate INTEGER NOT NULL,
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

            -- Import graph for "fake LSP" functionality
            CREATE TABLE IF NOT EXISTS file_dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL,
                target_import TEXT NOT NULL,
                resolved_file TEXT,
                UNIQUE(source_file, target_import)
            );

            CREATE INDEX IF NOT EXISTS idx_file_deps_source
            ON file_dependencies(source_file);

            CREATE INDEX IF NOT EXISTS idx_file_deps_resolved
            ON file_dependencies(resolved_file);

            -- Symbol index for 2-hop expansion
            CREATE TABLE IF NOT EXISTS symbol_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                symbol_type TEXT,
                chunk_id INTEGER,
                line_start INTEGER,
                line_end INTEGER,
                signature TEXT,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id)
            );

            CREATE INDEX IF NOT EXISTS idx_symbol_name
            ON symbol_index(symbol_name);

            CREATE INDEX IF NOT EXISTS idx_symbol_file
            ON symbol_index(file_path);
            """
        )

        for statement in (
            "ALTER TABLE chunk_embeddings ADD COLUMN backend TEXT NOT NULL DEFAULT 'hash'",
            "ALTER TABLE chunk_embeddings ADD COLUMN dimensions INTEGER NOT NULL DEFAULT 256",
            "ALTER TABLE chunk_embeddings ADD COLUMN model_name TEXT",
        ):
            try:
                conn.execute(statement)
            except sqlite3.OperationalError:
                pass

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


def get_cached_query_result(conn: sqlite3.Connection, cache_key: str, now_epoch: int) -> dict | None:
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
    existing = conn.execute("SELECT id FROM documents WHERE source = ?", (source,)).fetchone()

    if existing:
        document_id = int(existing["id"])
        old_chunks = conn.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,)).fetchall()
        for row in old_chunks:
            cid = int(row["id"])
            conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (cid,))
            conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

        conn.execute(
            """
            UPDATE documents
            SET raw_text = ?, cleaned_text = ?, updated_at = ?
            WHERE id = ?
            """,
            (raw_text, cleaned_text, now, document_id),
        )
    else:
        cursor = conn.execute(
            """
            INSERT INTO documents (source, raw_text, cleaned_text, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source, raw_text, cleaned_text, now, now),
        )
        document_id = int(cursor.lastrowid)

    if is_code_file(source):
        chunks = chunk_code(cleaned_text, source, chunk_size_words)
    else:
        chunks = chunk_text(cleaned_text, chunk_size_words, overlap_words)
    for idx, chunk in enumerate(chunks):
        cursor = conn.execute(
            """
            INSERT INTO chunks (document_id, chunk_index, text, token_estimate)
            VALUES (?, ?, ?, ?)
            """,
            (document_id, idx, chunk, estimate_tokens(chunk)),
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


def index_file_dependencies(
    conn: sqlite3.Connection,
    file_path: str,
    content: str,
    indexed_files: set[str],
) -> int:
    """Extract and store import dependencies for a file."""
    imports = extract_imports(content, file_path)
    count = 0

    for import_path in imports:
        resolved = resolve_import_to_file(import_path, file_path, indexed_files)
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO file_dependencies (source_file, target_import, resolved_file)
                VALUES (?, ?, ?)
                """,
                (file_path, import_path, resolved),
            )
            count += 1
        except Exception:
            continue

    return count


def index_symbols(
    conn: sqlite3.Connection,
    file_path: str,
    chunk_id: int,
    chunk_text: str,
) -> int:
    """Extract and store symbols from a chunk."""
    symbols = extract_symbols_from_chunk(chunk_text, file_path)
    count = 0

    for symbol in symbols:
        try:
            conn.execute(
                """
                INSERT INTO symbol_index (file_path, symbol_name, symbol_type, chunk_id, signature)
                VALUES (?, ?, ?, ?, ?)
                """,
                (file_path, symbol["name"], symbol["type"], chunk_id, symbol.get("signature")),
            )
            count += 1
        except Exception:
            continue

    return count


def extract_symbols_from_chunk(text: str, source: str) -> list[dict]:
    """Extract function/class symbols from a code chunk using AST or regex."""
    symbols: list[dict] = []
    ext = Path(source).suffix.lower()

    # Use regex patterns as fallback
    if ext == ".py":
        # Python: def func_name(...) or class ClassName
        for match in re.finditer(r"^(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "function",
                "signature": f"def {match.group(1)}({match.group(2)})",
            })
        for match in re.finditer(r"^class\s+(\w+)(?:\(([^)]*)\))?:", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "class",
                "signature": f"class {match.group(1)}",
            })
    elif ext in {".js", ".ts", ".tsx", ".jsx"}:
        # JS/TS: function name() or const name = () =>
        for match in re.finditer(r"(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "function",
                "signature": f"function {match.group(1)}({match.group(2)})",
            })
        for match in re.finditer(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "function",
                "signature": f"const {match.group(1)} = () =>",
            })
        for match in re.finditer(r"class\s+(\w+)(?:\s+extends\s+\w+)?", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "class",
                "signature": f"class {match.group(1)}",
            })
    elif ext == ".go":
        for match in re.finditer(r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(([^)]*)\)", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "function",
                "signature": f"func {match.group(1)}({match.group(2)})",
            })
        for match in re.finditer(r"type\s+(\w+)\s+struct", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "struct",
                "signature": f"type {match.group(1)} struct",
            })
    elif ext == ".rs":
        for match in re.finditer(r"(?:pub\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "function",
                "signature": f"fn {match.group(1)}({match.group(2)})",
            })
        for match in re.finditer(r"(?:pub\s+)?struct\s+(\w+)", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "struct",
                "signature": f"struct {match.group(1)}",
            })

    return symbols


def get_imported_files(conn: sqlite3.Connection, source_file: str) -> list[str]:
    """Get list of files imported by a source file."""
    rows = conn.execute(
        """
        SELECT DISTINCT resolved_file FROM file_dependencies
        WHERE source_file = ? AND resolved_file IS NOT NULL
        """,
        (source_file,),
    ).fetchall()
    return [str(row[0]) for row in rows]


def lookup_symbol_definition(conn: sqlite3.Connection, symbol_name: str, limit: int = 3) -> list[dict]:
    """Look up symbol definitions by name (for 2-hop expansion)."""
    rows = conn.execute(
        """
        SELECT s.file_path, s.symbol_type, s.signature, c.text
        FROM symbol_index s
        JOIN chunks c ON s.chunk_id = c.id
        WHERE s.symbol_name = ?
        LIMIT ?
        """,
        (symbol_name, limit),
    ).fetchall()

    return [
        {
            "file": row[0],
            "type": row[1],
            "signature": row[2],
            "definition": row[3][:500] if row[3] else None,  # Truncate long definitions
        }
        for row in rows
    ]


def expand_symbols_two_hop(
    conn: sqlite3.Connection,
    candidates: list[Candidate],
    max_expansions: int = 5,
) -> list[dict]:
    """Perform 2-hop symbol expansion on selected candidates.

    Extracts function calls from selected chunks and fetches their definitions.
    This simulates LSP "go to definition" functionality.
    """
    referenced_symbols: list[dict] = []
    seen_symbols: set[str] = set()

    for candidate in candidates[:3]:  # Only expand top 3 chunks
        calls = extract_function_calls(candidate.text)

        for call_name in calls:
            if call_name in seen_symbols:
                continue
            if len(referenced_symbols) >= max_expansions:
                break

            definitions = lookup_symbol_definition(conn, call_name, limit=1)
            if definitions:
                seen_symbols.add(call_name)
                referenced_symbols.append({
                    "symbol": call_name,
                    "from_chunk": candidate.source,
                    **definitions[0],
                })

    return referenced_symbols


def fetch_imported_context(
    conn: sqlite3.Connection,
    selected_sources: list[str],
    query: str,
    limit_per_file: int = 2,
) -> list[Candidate]:
    """Fetch additional context from files imported by selected sources."""
    additional_candidates: list[Candidate] = []
    seen_files: set[str] = set(selected_sources)

    for source in selected_sources:
        imported_files = get_imported_files(conn, source)

        for imported_file in imported_files:
            if imported_file in seen_files:
                continue
            seen_files.add(imported_file)

            # Do a quick FTS search in the imported file
            rows = conn.execute(
                """
                SELECT c.id, c.text, c.chunk_index, c.token_estimate, d.source
                FROM chunks_fts f
                JOIN chunks c ON f.rowid = c.id
                JOIN documents d ON c.document_id = d.id
                WHERE chunks_fts MATCH ? AND d.source = ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, imported_file, limit_per_file),
            ).fetchall()

            for row in rows:
                additional_candidates.append(
                    Candidate(
                        chunk_id=int(row[0]),
                        text=str(row[1]),
                        chunk_index=int(row[2]),
                        token_estimate=int(row[3]),
                        source=str(row[4]),
                        fts_rank=len(additional_candidates),
                    )
                )

    return additional_candidates
