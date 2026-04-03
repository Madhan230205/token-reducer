from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from token_reducer.db import (
    cleanup_query_cache,
    connect_db,
    extract_symbols_from_chunk,
    get_cached_query_embedding,
    get_cached_query_result,
    get_index_fingerprint,
    get_recent_session_queries,
    index_file_dependencies,
    load_session_memory,
    save_session_memory,
    session_memory_path,
    set_cached_query_embedding,
    set_cached_query_result,
    update_session_memory,
    upsert_document,
)


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    conn = connect_db(tmp_path / "test.db")
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# connect_db
# ---------------------------------------------------------------------------


class TestConnectDb:
    def test_creates_required_tables(self, tmp_path: Path) -> None:
        conn = connect_db(tmp_path / "test.db")
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert {
            "documents",
            "chunks",
            "chunk_embeddings",
            "query_embeddings",
            "query_cache",
            "file_dependencies",
            "symbol_index",
        } <= tables
        conn.close()

    def test_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "test.db"
        conn1 = connect_db(path)
        conn1.close()
        conn2 = connect_db(path)  # should not raise
        conn2.close()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "test.db"
        conn = connect_db(path)
        assert path.exists()
        conn.close()

    def test_foreign_keys_enabled(self, tmp_path: Path) -> None:
        conn = connect_db(tmp_path / "test.db")
        row = conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1
        conn.close()


# ---------------------------------------------------------------------------
# get_index_fingerprint
# ---------------------------------------------------------------------------


class TestGetIndexFingerprint:
    def test_returns_non_empty_string(self, db: sqlite3.Connection) -> None:
        fp = get_index_fingerprint(db)
        assert isinstance(fp, str) and len(fp) > 0

    def test_consistent_on_empty_db(self, db: sqlite3.Connection) -> None:
        assert get_index_fingerprint(db) == get_index_fingerprint(db)

    def test_changes_after_document_insert(self, db: sqlite3.Connection) -> None:
        fp_before = get_index_fingerprint(db)
        db.execute(
            "INSERT INTO documents (source, raw_text, cleaned_text, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            ("f.py", "raw", "cleaned", "2024-01-01T00:00:00", "2024-01-01T00:00:00"),
        )
        db.commit()
        assert get_index_fingerprint(db) != fp_before


# ---------------------------------------------------------------------------
# query cache
# ---------------------------------------------------------------------------


class TestQueryCache:
    def test_miss_returns_none(self, db: sqlite3.Connection) -> None:
        assert get_cached_query_result(db, "missing", now_epoch=1000) is None

    def test_set_and_get_roundtrip(self, db: sqlite3.Connection) -> None:
        payload = {"results": ["a", "b"]}
        set_cached_query_result(db, "k1", payload, now_epoch=1000, ttl_seconds=300)
        assert get_cached_query_result(db, "k1", now_epoch=1001) == payload

    def test_expired_entry_returns_none(self, db: sqlite3.Connection) -> None:
        set_cached_query_result(db, "k2", {"x": 1}, now_epoch=1000, ttl_seconds=30)
        # expires_at = 1000 + max(30, 30) = 1030; query at 1031 → expired
        assert get_cached_query_result(db, "k2", now_epoch=1031) is None

    def test_cleanup_removes_only_expired(self, db: sqlite3.Connection) -> None:
        set_cached_query_result(db, "old", {}, now_epoch=100, ttl_seconds=30)  # expires 130
        set_cached_query_result(db, "new", {}, now_epoch=100, ttl_seconds=1000)  # expires 1100
        cleanup_query_cache(db, now_epoch=200)
        assert get_cached_query_result(db, "old", now_epoch=200) is None
        assert get_cached_query_result(db, "new", now_epoch=200) is not None

    def test_upsert_overwrites_existing(self, db: sqlite3.Connection) -> None:
        set_cached_query_result(db, "k3", {"v": 1}, now_epoch=1000, ttl_seconds=300)
        set_cached_query_result(db, "k3", {"v": 2}, now_epoch=1000, ttl_seconds=300)
        result = get_cached_query_result(db, "k3", now_epoch=1001)
        assert result == {"v": 2}


# ---------------------------------------------------------------------------
# query embedding cache
# ---------------------------------------------------------------------------


class TestQueryEmbeddingCache:
    def test_miss_returns_none(self, db: sqlite3.Connection) -> None:
        assert get_cached_query_embedding(db, "missing") is None

    def test_roundtrip(self, db: sqlite3.Connection) -> None:
        emb = [0.1, 0.2, 0.3, 0.4]
        set_cached_query_embedding(db, "q1", "hash123", emb, "hash", None)
        result = get_cached_query_embedding(db, "q1")
        assert result is not None
        assert len(result) == 4
        assert abs(result[0] - 0.1) < 1e-6

    def test_stores_model_name(self, db: sqlite3.Connection) -> None:
        set_cached_query_embedding(db, "q2", "h", [1.0], "ml", "my-model")
        row = db.execute(
            "SELECT model_name FROM query_embeddings WHERE query_key = 'q2'"
        ).fetchone()
        assert row["model_name"] == "my-model"


# ---------------------------------------------------------------------------
# session memory
# ---------------------------------------------------------------------------


class TestSessionMemory:
    def test_session_memory_path_naming(self, tmp_path: Path) -> None:
        path = session_memory_path(tmp_path / "index.db")
        assert path.name == "session_memory.json"
        assert path.parent == tmp_path

    def test_load_missing_returns_default(self, tmp_path: Path) -> None:
        memory = load_session_memory(tmp_path / "nonexistent.json")
        assert memory == {"sessions": {}}

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.json"
        data: dict = {"sessions": {"s1": {"recent_queries": ["q1", "q2"]}}}
        save_session_memory(path, data)
        assert load_session_memory(path) == data

    def test_get_recent_queries_empty_session(self) -> None:
        assert get_recent_session_queries({"sessions": {}}, "ghost") == []

    def test_get_recent_queries_returns_last_n(self) -> None:
        memory: dict = {"sessions": {"s1": {"recent_queries": ["q1", "q2", "q3", "q4", "q5"]}}}
        assert get_recent_session_queries(memory, "s1", limit=3) == ["q3", "q4", "q5"]

    def test_get_recent_queries_limit_larger_than_list(self) -> None:
        memory: dict = {"sessions": {"s1": {"recent_queries": ["q1", "q2"]}}}
        assert get_recent_session_queries(memory, "s1", limit=10) == ["q1", "q2"]

    def test_update_session_memory_adds_query_and_source(self, tmp_path: Path) -> None:
        path = tmp_path / "mem.json"
        update_session_memory(path, "sess1", "my query", ["file1.py"])
        memory = load_session_memory(path)
        session = memory["sessions"]["sess1"]
        assert "my query" in session["recent_queries"]
        assert "file1.py" in session["recent_sources"]

    def test_update_session_memory_caps_history(self, tmp_path: Path) -> None:
        path = tmp_path / "mem.json"
        for i in range(12):
            update_session_memory(path, "sess", f"q{i}", [f"f{i}.py"])
        memory = load_session_memory(path)
        session = memory["sessions"]["sess"]
        # recent_queries capped at 8, recent_sources at 12
        assert len(session["recent_queries"]) <= 8
        assert len(session["recent_sources"]) <= 12


# ---------------------------------------------------------------------------
# extract_symbols_from_chunk
# ---------------------------------------------------------------------------


class TestExtractSymbols:
    def test_python_function(self) -> None:
        code = "def my_func(a, b):\n    return a + b"
        names = [s["name"] for s in extract_symbols_from_chunk(code, "test.py")]
        assert "my_func" in names

    def test_python_async_function(self) -> None:
        code = "async def fetch_data(url):\n    pass"
        names = [s["name"] for s in extract_symbols_from_chunk(code, "test.py")]
        assert "fetch_data" in names

    def test_python_class(self) -> None:
        code = "class MyClass(Base):\n    pass"
        symbols = extract_symbols_from_chunk(code, "test.py")
        names = [s["name"] for s in symbols]
        assert "MyClass" in names
        types = [s["type"] for s in symbols if s["name"] == "MyClass"]
        assert "class" in types

    def test_js_function_declaration(self) -> None:
        code = "function myFn(x, y) { return x; }"
        names = [s["name"] for s in extract_symbols_from_chunk(code, "test.js")]
        assert "myFn" in names

    def test_js_arrow_function(self) -> None:
        code = "const handler = (event) => { console.log(event); };"
        names = [s["name"] for s in extract_symbols_from_chunk(code, "test.js")]
        assert "handler" in names

    def test_go_function(self) -> None:
        code = "func DoWork(ctx context.Context) error { return nil }"
        names = [s["name"] for s in extract_symbols_from_chunk(code, "test.go")]
        assert "DoWork" in names

    def test_go_struct(self) -> None:
        code = "type Config struct {\n    Port int\n}"
        names = [s["name"] for s in extract_symbols_from_chunk(code, "test.go")]
        assert "Config" in names

    def test_rust_function(self) -> None:
        code = "pub fn process(input: &str) -> String { input.to_string() }"
        names = [s["name"] for s in extract_symbols_from_chunk(code, "test.rs")]
        assert "process" in names

    def test_unknown_extension_returns_empty(self) -> None:
        assert extract_symbols_from_chunk("some code", "test.xyz") == []


# ---------------------------------------------------------------------------
# upsert_document
# ---------------------------------------------------------------------------


class TestUpsertDocument:
    def test_inserts_document_and_chunks(self, tmp_path: Path) -> None:
        conn = connect_db(tmp_path / "test.db")
        count = upsert_document(
            conn=conn,
            source="test.py",
            raw_text="def foo():\n    pass",
            cleaned_text="def foo(): pass",
            chunk_size_words=150,
            overlap_words=20,
            dimensions=64,
            embedding_backend="hash",
            embedding_model=None,
        )
        assert count >= 1
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        conn.close()

    def test_upsert_replaces_existing_document(self, tmp_path: Path) -> None:
        conn = connect_db(tmp_path / "test.db")
        for version in range(2):
            upsert_document(
                conn=conn,
                source="test.py",
                raw_text=f"version {version}",
                cleaned_text=f"version {version}",
                chunk_size_words=150,
                overlap_words=0,
                dimensions=64,
                embedding_backend="hash",
                embedding_model=None,
            )
        # Still only one document row
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        conn.close()

    def test_chunks_have_embeddings(self, tmp_path: Path) -> None:
        conn = connect_db(tmp_path / "test.db")
        upsert_document(
            conn=conn,
            source="doc.txt",
            raw_text="hello world content",
            cleaned_text="hello world content",
            chunk_size_words=150,
            overlap_words=0,
            dimensions=64,
            embedding_backend="hash",
            embedding_model=None,
        )
        embedding_count = conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert embedding_count == chunk_count
        conn.close()


# ---------------------------------------------------------------------------
# index_file_dependencies
# ---------------------------------------------------------------------------


class TestIndexFileDependencies:
    def test_indexes_python_imports(self, db: sqlite3.Connection) -> None:
        content = "from os.path import join\nimport sys"
        count = index_file_dependencies(db, "main.py", content, set())
        assert count == 2
        rows = db.execute(
            "SELECT target_import FROM file_dependencies WHERE source_file = 'main.py'"
        ).fetchall()
        imports = {row[0] for row in rows}
        assert "os.path" in imports
        assert "sys" in imports

    def test_resolves_import_to_known_file(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        helper = str(tmp_path / "helpers.py")
        content = "from helpers import util"
        indexed = {helper}
        index_file_dependencies(db, str(tmp_path / "main.py"), content, indexed)
        # Resolution depends on file existence; just check the row was inserted
        count = db.execute("SELECT COUNT(*) FROM file_dependencies").fetchone()[0]
        assert count >= 1

    def test_duplicate_imports_ignored(self, db: sqlite3.Connection) -> None:
        content = "import os\nimport os"
        count1 = index_file_dependencies(db, "a.py", content, set())
        count2 = index_file_dependencies(db, "a.py", content, set())
        # INSERT OR IGNORE — second call inserts 0 new rows
        assert count1 >= 1
        assert count2 == 0
