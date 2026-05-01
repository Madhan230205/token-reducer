"""Intent-aware structural scoring and Claude blend ranking."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .intent import IntentType
from .models import Candidate
from .retriever import bm25_to_score


def _document_mtimes(conn: sqlite3.Connection, sources: list[str]) -> dict[str, float]:
    if not sources:
        return {}
    uniq = list(set(sources))
    ph = ",".join("?" * len(uniq))
    rows = conn.execute(
        f"SELECT source, file_mtime, updated_at FROM documents WHERE source IN ({ph})",
        uniq,
    ).fetchall()
    out: dict[str, float] = {}
    now = time.time()
    for row in rows:
        src = str(row["source"])
        mt = row["file_mtime"]
        if mt is not None:
            out[src] = float(mt)
        else:
            out[src] = now
    return out


def structural_score(
    candidate: Candidate,
    query: str,
    intent: IntentType,
    mtime_map: dict[str, float],
    now_ts: float,
) -> float:
    q = query.lower()
    text = candidate.text.lower()
    path = candidate.source.lower()
    name = Path(candidate.source).name.lower()
    s = 0.0

    if intent == "bug_fix":
        if any(
            k in q or k in text for k in ("error", "exception", "traceback", "fail", "bug", "panic")
        ):
            s += 0.28
        if any(k in text for k in ("raise ", "except ", "error(", "catch ", "panic!")):
            s += 0.22
        mt = mtime_map.get(candidate.source)
        if mt is not None:
            age_days = max(0.0, (now_ts - mt) / 86400.0)
            s += 0.35 * max(0.0, 1.0 - min(1.0, age_days / 30.0))
    elif intent == "feature_add":
        if any(
            k in path for k in ("main", "app", "index", "route", "api", "handler", "server", "cli")
        ):
            s += 0.32
        if any(
            k in text
            for k in ("export ", "router", "handler", "def main", "public static void main")
        ):
            s += 0.22
        if name in {"main.py", "app.py", "index.ts", "index.js", "lib.rs", "main.go"}:
            s += 0.2
    elif intent == "explain_code":
        sig_lines = sum(
            1 for line in candidate.text.splitlines()[:20] if "(" in line and ")" in line
        )
        if sig_lines >= 2:
            s += 0.25
        if "class " in text or "def " in text or "function " in text or "fn " in text:
            s += 0.35
        if 5 <= len(candidate.text.split()) <= 400:
            s += 0.2
    elif intent == "refactor":
        wc = len(candidate.text.split())
        if wc >= 80:
            s += 0.35
        if candidate.text.count("\n") >= 12:
            s += 0.25
        if any(k in q for k in ("refactor", "extract", "split", "class", "module")):
            s += 0.2
    else:  # navigation
        qtok = set(q.replace("/", " ").split())
        ntok = set(name.replace(".", "_").split("_"))
        if qtok & ntok:
            s += 0.45
        if any(
            Path(candidate.source).suffix.lower().endswith(ext)
            for ext in (".py", ".ts", ".rs", ".go")
        ):
            s += 0.15

    return min(1.0, s)


def _minmax(vals: list[float]) -> list[float]:
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def apply_claude_intent_rerank(
    conn: sqlite3.Connection,
    candidates: list[Candidate],
    query: str,
    intent: IntentType,
) -> list[Candidate]:
    """Re-score candidates: 0.4 BM25 + 0.3 semantic + 0.3 structural (intent-aware)."""
    if not candidates:
        return candidates
    now_ts = time.time()
    mtime_map = _document_mtimes(conn, [c.source for c in candidates])
    bm = [bm25_to_score(c.bm25_score) for c in candidates]
    vec = [
        max(0.0, min(1.0, max(float(c.vector_score), float(c.fts_score) * 0.85)))
        for c in candidates
    ]
    struct = [structural_score(c, query, intent, mtime_map, now_ts) for c in candidates]
    nb, nv, ns = _minmax(bm), _minmax(vec), _minmax(struct)
    for i, c in enumerate(candidates):
        c.structural_score = struct[i]
        c.final_score = 0.4 * nb[i] + 0.3 * nv[i] + 0.3 * ns[i]
    ranked = sorted(candidates, key=lambda x: x.final_score, reverse=True)
    return ranked
