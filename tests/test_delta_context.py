"""Tests for stateful delta context (anti-amnesia) omission."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from token_reducer.chunker import clean_text
from token_reducer.db import connect_db, load_session_memory, upsert_document
from token_reducer.delta_context import (
    active_context_signature,
    load_active_fingerprints,
    merge_active_context_entries,
    partition_redundant_candidates,
    persist_delivered_fingerprints,
)
from token_reducer.models import Candidate


def _candidate(conn: sqlite3.Connection, source: str, chunk_index: int) -> Candidate:
    row = conn.execute(
        """
        SELECT c.id, c.chunk_index, c.text, c.token_estimate, d.source
        FROM chunks c JOIN documents d ON c.document_id = d.id
        WHERE d.source = ? AND c.chunk_index = ?
        """,
        (source, chunk_index),
    ).fetchone()
    assert row is not None
    return Candidate(
        chunk_id=int(row["id"]),
        source=str(row["source"]),
        chunk_index=int(row["chunk_index"]),
        text=str(row["text"]),
        token_estimate=int(row["token_estimate"]),
        final_score=1.0,
    )


def test_partition_omits_when_active_matches_snapshot(tmp_path: Path) -> None:
    db_path = tmp_path / "t.db"
    conn = connect_db(db_path)
    try:
        src = tmp_path / "a.py"
        src.write_text("def foo():\n    return 1\n", encoding="utf-8")
        raw = src.read_text(encoding="utf-8")
        upsert_document(
            conn=conn,
            source=str(src),
            raw_text=raw,
            cleaned_text=clean_text(raw),
            chunk_size_words=80,
            overlap_words=10,
            dimensions=256,
            embedding_backend="hash",
            embedding_model=None,
        )
        conn.commit()
        c0 = _candidate(conn, str(src), 0)
        memory: dict = {"sessions": {}}
        merge_active_context_entries(
            memory,
            "s1",
            [
                {
                    "chunk_id": c0.chunk_id,
                    "source": c0.source,
                    "chunk_index": c0.chunk_index,
                    "file_hash": "dummy",
                    "file_mtime": None,
                }
            ],
        )
        kept, omitted = partition_redundant_candidates(
            conn, load_active_fingerprints(memory, "s1"), [c0]
        )
        assert len(kept) == 1 and not omitted

        row = conn.execute(
            "SELECT file_hash, file_mtime FROM documents WHERE source = ?", (str(src),)
        ).fetchone()
        assert row and row["file_hash"]
        fp = {
            "chunk_id": c0.chunk_id,
            "source": c0.source,
            "chunk_index": c0.chunk_index,
            "file_hash": str(row["file_hash"]),
            "file_mtime": float(row["file_mtime"]) if row["file_mtime"] is not None else None,
        }
        memory2: dict = {"sessions": {}}
        merge_active_context_entries(memory2, "s2", [fp])
        kept2, omitted2 = partition_redundant_candidates(
            conn, load_active_fingerprints(memory2, "s2"), [c0]
        )
        assert not kept2 and len(omitted2) == 1
        assert omitted2[0].status == "omitted_redundant"
    finally:
        conn.close()


def test_persist_roundtrip_and_signature(tmp_path: Path) -> None:
    mem_path = tmp_path / "session_memory.json"
    persist_delivered_fingerprints(
        mem_path,
        "sess",
        [
            {
                "chunk_id": 7,
                "source": "/x.py",
                "chunk_index": 0,
                "file_hash": "abc",
                "file_mtime": 1.5,
            }
        ],
    )
    blob = load_session_memory(mem_path)
    assert active_context_signature(blob, "sess")
    assert len(load_active_fingerprints(blob, "sess")) == 1
