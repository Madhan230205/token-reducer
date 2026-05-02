"""Shared run memory + deterministic task decomposition (no LLM)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..models import Candidate

_PATH_RE = re.compile(
    r"\b((?:[\w.\-]+[/\\])+[\w.\-]+\.(?:py|ts|tsx|js|jsx|mjs|cjs|go|rs|java|kt|rb|php|cs|swift|md|yaml|yml|toml|json))\b",
    re.I,
)

_STOP = frozenset(
    "the a an and or for to of in on at by is are was were be been being it this that these those "
    "with from as if when into over out up down all any some not no yes how what why where which "
    "who can could should would will just like into about into than then them their there".split()
)


def extract_focus_paths(prompt: str, *, limit: int = 6) -> list[str]:
    """File-like path hints from the user query (basename deduped)."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _PATH_RE.finditer(prompt or ""):
        raw = (m.group(1) or "").strip().strip("\"'`")
        if len(raw) < 4:
            continue
        norm = raw.replace("\\", "/")
        base = norm.split("/")[-1].lower()
        if base in seen:
            continue
        seen.add(base)
        out.append(norm)
        if len(out) >= limit:
            break
    return out


def extract_focus_terms(prompt: str, *, limit: int = 8) -> list[str]:
    """Short lexical hooks for cross-agent ranking (no stopwords)."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", (prompt or ""))
    out: list[str] = []
    seen: set[str] = set()
    for w in words:
        low = w.lower()
        if low in _STOP or low in seen:
            continue
        seen.add(low)
        out.append(w)
        if len(out) >= limit:
            break
    return out


def decompose_tasks(prompt: str, structured: dict[str, Any]) -> list[str]:
    """2–4 coordination steps keyed off legacy intent (deterministic)."""
    legacy = str(structured.get("legacy_intent", "explain_code"))
    coarse = str(structured.get("type", "analysis"))
    if coarse == "chat":
        return [
            "Address the user's immediate ask in plain language.",
            "Use attached material only if it clearly helps.",
        ]
    plans: dict[str, list[str]] = {
        "bug_fix": [
            "Isolate symptoms, errors, and failing assumptions from the query.",
            "Cross-check candidate code against guards, validation, and edge paths.",
            "Prefer the smallest change that restores the described behavior.",
        ],
        "feature_add": [
            "Identify the user-facing surface and extension points they care about.",
            "Align snippets with routes, handlers, and data flow implied by the ask.",
            "Keep new behavior scoped to what the query actually requests.",
        ],
        "navigation": [
            "Resolve symbols and paths the user named or implied.",
            "Prefer definitions and exports over incidental references.",
        ],
        "refactor": [
            "Find structural duplication or coupling visible in the snippets.",
            "Preserve public contracts while improving internal shape.",
        ],
        "explain_code": [
            "Ground explanation in behavior visible in the excerpts.",
            "Call out gaps explicitly instead of guessing beyond the text.",
        ],
    }
    base = list(plans.get(legacy, plans["explain_code"]))
    q = (prompt or "").strip()
    if len(q.split()) > 24:
        base.insert(1, "Prioritize the densest technical detail the user supplied.")
    return base[:4]


def build_run_memory(prompt: str, structured: dict[str, Any]) -> dict[str, Any]:
    """Shared dict read by downstream agents in the same pass."""
    return {
        "decomposition": decompose_tasks(prompt, structured),
        "focus_paths": extract_focus_paths(prompt),
        "focus_terms": extract_focus_terms(prompt),
    }


def _path_hit(c: Candidate, paths: list[str]) -> bool:
    cnorm = str(c.source).replace("\\", "/").lower()
    for p in paths:
        if not isinstance(p, str):
            continue
        pn = p.replace("\\", "/").lower()
        if pn in cnorm or Path(p).name.lower() in cnorm:
            return True
    return False


def merge_subtask_streams(chunks: list[Candidate], run_memory: dict[str, Any]) -> list[Candidate]:
    """Interleave path-aligned, term-aligned, and remaining chunks (same multiset, new order)."""
    paths = [p for p in (run_memory.get("focus_paths") or []) if isinstance(p, str)]
    terms = [t.lower() for t in (run_memory.get("focus_terms") or []) if isinstance(t, str)]
    if not paths and not terms:
        return chunks
    ranked = sorted(chunks, key=lambda c: float(c.final_score), reverse=True)
    path_hits: list[Candidate] = []
    path_ids: set[int] = set()
    for c in ranked:
        if paths and _path_hit(c, paths):
            path_hits.append(c)
            path_ids.add(c.chunk_id)
    term_hits: list[Candidate] = []
    term_ids: set[int] = set()
    for c in ranked:
        if c.chunk_id in path_ids:
            continue
        if terms:
            blob = (c.text + " " + c.source).lower()
            if any(t in blob for t in terms):
                term_hits.append(c)
                term_ids.add(c.chunk_id)
    rest = [c for c in ranked if c.chunk_id not in path_ids and c.chunk_id not in term_ids]
    seen: set[int] = set()
    out: list[Candidate] = []
    ip = it = ir = 0

    def _take(bucket: list[Candidate], idx: int) -> tuple[Candidate | None, int]:
        while idx < len(bucket):
            c = bucket[idx]
            idx += 1
            if c.chunk_id not in seen:
                return c, idx
        return None, idx

    while len(out) < len(ranked):
        progressed = False
        c, ip = _take(path_hits, ip)
        if c is not None:
            out.append(c)
            seen.add(c.chunk_id)
            progressed = True
        c, it = _take(term_hits, it)
        if c is not None:
            out.append(c)
            seen.add(c.chunk_id)
            progressed = True
        c, ir = _take(rest, ir)
        if c is not None:
            out.append(c)
            seen.add(c.chunk_id)
            progressed = True
        if not progressed:
            for c in ranked:
                if c.chunk_id not in seen:
                    out.append(c)
                    seen.add(c.chunk_id)
            break
    return out


def diversify_chunk_order(chunks: list[Candidate], prompt: str) -> list[Candidate]:
    """Rare, deterministic reorder: surface one mid-list chunk from an underrepresented file."""
    if len(chunks) < 7:
        return chunks
    out = list(chunks)
    top_stems = {Path(c.source).stem for c in out[:3]}
    slot = 4 + (abs(hash((prompt or "")[:200])) % 4)
    if slot >= len(out):
        return out
    cand = out[slot]
    stem = Path(cand.source).stem
    if stem in top_stems:
        return out
    out.pop(slot)
    out.insert(min(2, len(out)), cand)
    return out
