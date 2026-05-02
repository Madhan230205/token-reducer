"""Bounded caller/callee fanout: adjacent chunks, role neighbors, symbol meta — no full AST."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import Candidate
from .repo_map import RepoMap


def _candidate_from_row(row: sqlite3.Row, *, final_score: float) -> Candidate:
    return Candidate(
        chunk_id=int(row["chunk_id"]),
        source=str(row["source"]),
        chunk_index=int(row["chunk_index"]),
        text=str(row["text"]),
        token_estimate=int(row["token_estimate"] or 1),
        bm25_score=None,
        fts_rank=None,
        final_score=final_score,
    )


def _neighbor_sources(
    repo_map: RepoMap,
    seed_sources: list[str],
    *,
    include_callers: bool,
    include_callees: bool,
    cap: int = 8,
) -> frozenset[str]:
    """Same-directory service (caller-ish) and util/helper (callee-ish) files."""
    if not seed_sources:
        return frozenset()
    parents = {str(Path(s).parent) for s in seed_sources}
    out: list[str] = []
    if include_callers:
        for src in repo_map.service_sources:
            if str(Path(src).parent) in parents:
                out.append(src)
    if include_callees:
        for src in repo_map.utility_sources | repo_map.helper_sources:
            if str(Path(src).parent) in parents:
                out.append(src)
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
        if len(uniq) >= cap:
            break
    return frozenset(uniq)


def fetch_adjacent_chunks(
    conn: sqlite3.Connection,
    source: str,
    chunk_index: int,
    *,
    radius: int = 1,
) -> list[Candidate]:
    rows = conn.execute(
        """
        SELECT c.id AS chunk_id, c.text, c.chunk_index, c.token_estimate, d.source
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.source = ? AND c.chunk_index BETWEEN ? AND ?
        ORDER BY c.chunk_index
        """,
        (source, chunk_index - radius, chunk_index + radius),
    ).fetchall()
    out: list[Candidate] = []
    for r in rows:
        idx = int(r["chunk_index"])
        gap = abs(idx - chunk_index)
        base = 0.13 + 0.02 * max(0, radius - gap)
        out.append(_candidate_from_row(r, final_score=min(0.21, base)))
    return out


def fetch_head_chunks_for_sources(
    conn: sqlite3.Connection,
    sources: frozenset[str],
    *,
    per_source: int = 1,
    max_total: int = 10,
) -> list[Candidate]:
    acc: list[Candidate] = []
    for i, src in enumerate(list(sources)[:max_total]):
        rows = conn.execute(
            """
            SELECT c.id AS chunk_id, c.text, c.chunk_index, c.token_estimate, d.source
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.source = ?
            ORDER BY c.chunk_index
            LIMIT ?
            """,
            (src, per_source),
        ).fetchall()
        for r in rows:
            acc.append(_candidate_from_row(r, final_score=0.12 + 0.003 * i))
    return acc


def fetch_chunks_matching_meta_tokens(
    conn: sqlite3.Connection,
    tokens: frozenset[str],
    *,
    limit_per_token: int = 2,
    max_total: int = 10,
) -> list[Candidate]:
    """meta_json contains symbol-ish keys; LIKE is bounded by few tokens."""
    if not tokens:
        return []
    acc: list[Candidate] = []
    seen: set[int] = set()
    for tok in list(tokens)[:5]:
        if len(tok) < 3:
            continue
        like = f"%{tok}%"
        rows = conn.execute(
            """
            SELECT c.id AS chunk_id, c.text, c.chunk_index, c.token_estimate, d.source, c.meta_json
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.meta_json IS NOT NULL AND c.meta_json != '' AND c.meta_json LIKE ?
            LIMIT ?
            """,
            (like, limit_per_token),
        ).fetchall()
        for r in rows:
            cid = int(r["chunk_id"])
            if cid in seen:
                continue
            seen.add(cid)
            raw = r["meta_json"]
            bonus = 0.01
            if isinstance(raw, str):
                try:
                    data = json.loads(raw)
                    if isinstance(data, dict) and str(data.get("symbol_name", "")) == tok:
                        bonus = 0.03
                except (json.JSONDecodeError, TypeError):
                    pass
            acc.append(_candidate_from_row(r, final_score=0.14 + bonus))
            if len(acc) >= max_total:
                return acc
    return acc


def expand_context_candidates(
    conn: sqlite3.Connection,
    scored_top: list[Candidate],
    repo_map: RepoMap | None,
    must_keep: frozenset[str],
    *,
    include_callers: bool,
    include_callees: bool,
    max_extra: int = 22,
) -> list[Candidate]:
    """Return new candidates (not in scored_top) — adjacent + role neighbors + meta symbol hits."""
    if max_extra <= 0:
        return []
    seen: set[int] = {c.chunk_id for c in scored_top}
    out: list[Candidate] = []

    def push(c: Candidate) -> None:
        nonlocal out
        if c.chunk_id in seen:
            return
        if len(out) >= max_extra:
            return
        seen.add(c.chunk_id)
        out.append(c)

    seeds = scored_top[:6]
    for s in seeds:
        for c in fetch_adjacent_chunks(conn, s.source, s.chunk_index, radius=1):
            push(c)
            if len(out) >= max_extra:
                return out

    seed_sources = [c.source for c in seeds]
    if repo_map is not None and (include_callers or include_callees):
        extra_src = _neighbor_sources(
            repo_map,
            seed_sources,
            include_callers=include_callers,
            include_callees=include_callees,
            cap=8,
        )
        for c in fetch_head_chunks_for_sources(conn, extra_src, per_source=1, max_total=8):
            push(c)
            if len(out) >= max_extra:
                return out

    if must_keep:
        for c in fetch_chunks_matching_meta_tokens(conn, must_keep, limit_per_token=2, max_total=8):
            push(c)
            if len(out) >= max_extra:
                break

    return out
