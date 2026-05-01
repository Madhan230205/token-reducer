"""Incremental index updates (hash-aware)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from .db import detect_file_changes, index_corpus, remove_documents_by_source


def get_changed_files(
    conn: sqlite3.Connection,
    file_paths: list[Path],
) -> dict[str, list[Path] | list[str]]:
    new_files, modified_files, deleted_sources = detect_file_changes(conn, file_paths)
    return {"new": new_files, "modified": modified_files, "deleted_sources": deleted_sources}


def update_index_incremental(
    conn: sqlite3.Connection,
    file_paths: Iterable[Path],
    chunk_size_words: int,
    overlap_words: int,
    dimensions: int,
    embedding_backend: str,
    embedding_model: str | None,
) -> dict[str, int]:
    paths = list(file_paths)
    ch = get_changed_files(conn, paths)
    deleted_sources = ch["deleted_sources"]
    if deleted_sources:
        remove_documents_by_source(conn, deleted_sources)
    to_index = list(ch["new"]) + list(ch["modified"])
    stats = {"removed": len(deleted_sources), "indexed_files": 0, "chunks": 0}
    if to_index:
        r = index_corpus(
            conn=conn,
            file_paths=to_index,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            dimensions=dimensions,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
        )
        stats["indexed_files"] = r["files_indexed"]
        stats["chunks"] = r["chunks_indexed"]
    conn.commit()
    return stats
