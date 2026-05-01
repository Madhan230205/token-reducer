"""Stateful delta context: omit chunks already delivered in-session when files unchanged."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Candidate, OmittedRedundantEntry, hash_text


@dataclass(frozen=True)
class DocumentFingerprint:
    source: str
    file_hash: str | None
    file_mtime: float | None


def fetch_document_fingerprint_for_chunk(
    conn: sqlite3.Connection, chunk_id: int
) -> DocumentFingerprint | None:
    row = conn.execute(
        """
        SELECT d.source AS source, d.file_hash AS file_hash, d.file_mtime AS file_mtime
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.id = ?
        """,
        (chunk_id,),
    ).fetchone()
    if not row:
        return None
    src = str(row["source"])
    fh = row["file_hash"]
    mt = row["file_mtime"]
    file_hash = str(fh) if fh is not None else None
    file_mtime = float(mt) if mt is not None else None
    return DocumentFingerprint(source=src, file_hash=file_hash, file_mtime=file_mtime)


def _mtime_key(m: float | None) -> str:
    if m is None:
        return ""
    return f"{m:.6f}"


def active_fingerprint_key(
    source: str, chunk_index: int, file_hash: str | None, mtime: float | None
) -> str:
    return f"{source}\t{chunk_index}\t{file_hash or ''}\t{_mtime_key(mtime)}"


def active_context_count(memory: dict[str, Any], session_id: str) -> int:
    return len(load_active_fingerprints(memory, session_id))


def load_active_fingerprints(memory: dict[str, Any], session_id: str) -> list[dict[str, Any]]:
    sessions = memory.get("sessions", {})
    if not isinstance(sessions, dict):
        return []
    session = sessions.get(session_id, {})
    if not isinstance(session, dict):
        return []
    raw = session.get("active_context", [])
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict) and "source" in item and "chunk_index" in item:
            out.append(item)
    return out


def active_context_signature(memory: dict[str, Any], session_id: str) -> str:
    fps = load_active_fingerprints(memory, session_id)
    keys = sorted(
        active_fingerprint_key(
            str(x.get("source", "")),
            int(x.get("chunk_index", -1)),
            str(x["file_hash"]) if x.get("file_hash") is not None else None,
            float(x["file_mtime"]) if x.get("file_mtime") is not None else None,
        )
        for x in fps
    )
    return hash_text(json.dumps(keys, sort_keys=True))


def _active_has_match(
    active: list[dict[str, Any]],
    source: str,
    chunk_index: int,
    file_hash: str | None,
    file_mtime: float | None,
) -> bool:
    if file_hash is None:
        return False
    want = active_fingerprint_key(source, chunk_index, file_hash, file_mtime)
    for item in active:
        if not isinstance(item, dict):
            continue
        got = active_fingerprint_key(
            str(item.get("source", "")),
            int(item.get("chunk_index", -1)),
            str(item["file_hash"]) if item.get("file_hash") is not None else None,
            float(item["file_mtime"]) if item.get("file_mtime") is not None else None,
        )
        if got == want:
            return True
    return False


def partition_redundant_candidates(
    conn: sqlite3.Connection,
    active_fingerprints: list[dict[str, Any]],
    candidates: list[Candidate],
) -> tuple[list[Candidate], list[OmittedRedundantEntry]]:
    """Keep candidates not already in active context with same file snapshot; omit the rest."""
    kept: list[Candidate] = []
    omitted: list[OmittedRedundantEntry] = []
    for c in candidates:
        fp = fetch_document_fingerprint_for_chunk(conn, c.chunk_id)
        if fp is None or fp.file_hash is None:
            kept.append(c)
            continue
        if _active_has_match(
            active_fingerprints, c.source, c.chunk_index, fp.file_hash, fp.file_mtime
        ):
            omitted.append(
                OmittedRedundantEntry(
                    chunk_id=c.chunk_id,
                    source=c.source,
                    chunk_index=c.chunk_index,
                    reason="Chunk already in active context (file hash/mtime unchanged)",
                )
            )
        else:
            kept.append(c)
    return kept, omitted


def fingerprint_dict_for_chunk(conn: sqlite3.Connection, c: Candidate) -> dict[str, Any] | None:
    fp = fetch_document_fingerprint_for_chunk(conn, c.chunk_id)
    if fp is None or fp.file_hash is None:
        return None
    return {
        "chunk_id": c.chunk_id,
        "source": c.source,
        "chunk_index": c.chunk_index,
        "file_hash": fp.file_hash,
        "file_mtime": fp.file_mtime,
    }


def merge_active_context_entries(
    memory: dict[str, Any],
    session_id: str,
    new_entries: list[dict[str, Any]],
    max_entries: int = 500,
) -> dict[str, Any]:
    """Prepend delivered chunk fingerprints; dedupe by snapshot key; cap list size."""
    sessions = memory.setdefault("sessions", {})
    session = sessions.setdefault(session_id, {"recent_queries": [], "recent_sources": []})
    active = session.setdefault("active_context", [])
    if not isinstance(active, list):
        active = []
        session["active_context"] = active

    def _norm(d: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(d, dict) or d.get("file_hash") is None:
            return None
        return {
            "chunk_id": int(d["chunk_id"]),
            "source": str(d["source"]),
            "chunk_index": int(d["chunk_index"]),
            "file_hash": str(d["file_hash"]),
            "file_mtime": float(d["file_mtime"]) if d.get("file_mtime") is not None else None,
        }

    def _key(d: dict[str, Any]) -> str:
        return active_fingerprint_key(
            str(d["source"]),
            int(d["chunk_index"]),
            str(d["file_hash"]),
            float(d["file_mtime"]) if d.get("file_mtime") is not None else None,
        )

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in new_entries:
        n = _norm(item)
        if n is None:
            continue
        k = _key(n)
        if k in seen:
            continue
        seen.add(k)
        merged.append(n)
    for item in active:
        n = _norm(item)
        if n is None:
            continue
        k = _key(n)
        if k in seen:
            continue
        seen.add(k)
        merged.append(n)
        if len(merged) >= max_entries:
            break

    session["active_context"] = merged[:max_entries]
    sessions[session_id] = session
    memory["sessions"] = sessions
    return memory


def persist_delivered_fingerprints(
    memory_path: Path,
    session_id: str,
    new_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    from .db import load_session_memory, save_session_memory

    memory = load_session_memory(memory_path)
    merge_active_context_entries(memory, session_id, new_entries)
    save_session_memory(memory_path, memory)
    return memory
