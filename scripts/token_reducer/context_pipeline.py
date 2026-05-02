"""Single dominant context pipeline: intent → retrieve → rerank → subagents → compress → expand.

**Cache-miss execution MUST go through** :func:`process_prompt` (see :func:`run_retrieval_pipeline`).
Subagents are coordinator → merge_streams → filter → ranking → variance → fusion → budget (deterministic, pre-compression).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import sqlite3

from .chunker import estimate_tokens
from .compressor import trim_to_words
from .intent import StructuredIntent, detect_intent, structured_intent_to_dict
from .models import Candidate
from .ranking import _document_mtimes, structural_score
from .retriever import overlap_ratio
from .subagents.policy import effective_chunk_token_budget

# Aggressive early prune before expensive rerank / structural scoring (sharp pipeline).
EARLY_MERGE_CAP = 50
EARLY_PRUNE_K = 15


def early_prune_candidates(
    pool: list[Candidate],
    *,
    merge_cap: int | None = None,
    prune_k: int | None = None,
) -> list[Candidate]:
    """Cap merged pool by score, then keep top ``prune_k`` (autonomous strategy may override)."""
    if not pool:
        return pool
    cap = int(merge_cap if merge_cap is not None else EARLY_MERGE_CAP)
    k = int(prune_k if prune_k is not None else EARLY_PRUNE_K)
    ranked = sorted(pool, key=lambda c: float(c.final_score), reverse=True)
    capped = ranked[: min(cap, len(ranked))]
    if len(capped) <= k:
        return capped
    return capped[:k]


def _final_bullet_token_estimate(bullets: list[str]) -> int:
    return sum(estimate_tokens(b) for b in bullets)


def ensure_bullets_within_token_budget(bullets: list[str], budget: int) -> list[str]:
    """Drop lowest-priority bullets (end of list) until under budget; never grows text."""
    if budget <= 0:
        return []
    out = list(bullets)
    while out and _final_bullet_token_estimate(out) > budget:
        out.pop()
    if out:
        return out
    if not bullets:
        return []
    # Hard clamp single bullet
    words = max(12, min(len(bullets[0].split()), budget * 2 // 3))
    return [trim_to_words(bullets[0], max_words=words)]


def _minmax(vals: list[float]) -> list[float]:
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def rerank_chunks(
    chunks: list[Candidate],
    query: str,
    intent: StructuredIntent | dict[str, Any],
    *,
    conn: sqlite3.Connection | None = None,
    top_n: int | None = None,
) -> list[Candidate]:
    """Layer after lexical/vector merge: semantic + keyword + recency (+ light structural).

    ``semantic_similarity`` uses the merged retrieval signal (``final_score``,
    ``vector_score``, ``fts_score``). ``keyword_score`` is query/chunk token overlap.
    ``recency_score`` uses document mtime when ``conn`` is provided.

    Formula: ``0.6 * semantic + 0.3 * keyword + 0.1 * recency``, blended with a small
    structural term so bug-fix / navigation heuristics from :mod:`ranking` still apply.
    """
    if not chunks:
        return chunks

    if isinstance(intent, dict):
        legacy = intent.get("legacy_intent", "explain_code")
        compression_level = str(intent.get("compression_level", "medium"))
    else:
        legacy = intent["legacy_intent"]
        compression_level = intent["compression_level"]

    now_ts = time.time()
    sources = [c.source for c in chunks]
    mtime_map: dict[str, float] = {}
    if conn is not None:
        mtime_map = _document_mtimes(conn, sources)

    sem_raw: list[float] = []
    for c in chunks:
        sem_raw.append(
            max(float(c.final_score), float(c.vector_score), float(c.fts_score), 1e-6)
        )
    kw_raw = [overlap_ratio(query, c.text) for c in chunks]
    rec_raw: list[float] = []
    for c in chunks:
        mt = mtime_map.get(c.source)
        if mt is not None:
            age_days = max(0.0, (now_ts - mt) / 86400.0)
            rec_raw.append(max(0.0, 1.0 - min(1.0, age_days / 120.0)))
        else:
            rec_raw.append(0.45)

    sem_n = _minmax(sem_raw)
    kw_n = _minmax(kw_raw)
    rec_n = _minmax(rec_raw)

    struct_raw = [
        structural_score(c, query, legacy, mtime_map, now_ts) for c in chunks  # type: ignore[arg-type]
    ]
    struct_n = _minmax(struct_raw)

    level_w = {"high": 1.04, "medium": 1.0, "low": 0.96}.get(compression_level, 1.0)

    for i, c in enumerate(chunks):
        base = 0.6 * sem_n[i] + 0.3 * kw_n[i] + 0.1 * rec_n[i]
        mix = 0.82 * base + 0.18 * struct_n[i]
        c.structural_score = float(struct_raw[i])
        c.final_score = float(mix * level_w)

    ranked = sorted(chunks, key=lambda x: x.final_score, reverse=True)
    if top_n is not None and top_n > 0:
        return ranked[:top_n]
    return ranked


def expand_context_if_needed(
    bullets: list[str],
    intent: StructuredIntent | dict[str, Any],
    conn: sqlite3.Connection | None,
    ranked_pool: list[Candidate],
) -> list[str]:
    """Post-compression refinement: only when output is thin and intent is code/analysis."""
    itype = intent["type"] if isinstance(intent, dict) else intent["type"]
    if itype == "chat":
        return bullets
    if len(bullets) >= 2:
        return bullets
    if not ranked_pool or conn is None:
        return bullets
    top = ranked_pool[0]
    head = "\n".join(top.text.splitlines()[:4]).strip()
    if not head:
        return bullets
    rel = Path(top.source).name
    extra = f"(context_pipeline) adjacent signal [{rel}#chunk-{top.chunk_index}]: {head[:320]}"
    return [*bullets, extra]


def inject_context(prompt: str, context: str) -> str:
    """Prefix optimized context ahead of the user prompt (hook / demo friendly)."""
    ctx = (context or "").strip()
    if not ctx:
        return prompt
    return f"{ctx}\n\n---\n\n{prompt}"


def process_prompt(
    prompt: str,
    state: dict[str, Any] | None = None,
    *,
    debug: bool = False,
) -> dict[str, Any]:
    """Run the dominant pipeline when a live :class:`ContextRunState` is supplied.

    ``state`` may contain:

    - ``context_run_state``: a :class:`ContextRunState` instance (mutated in place).
    - ``orchestrator``: a :class:`ContextPipelineOrchestrator` wrapping that state.

    Without DB state, returns structured intent and empty context (backward-safe).
    """
    from .orchestrator import ContextPipelineOrchestrator, ContextRunState

    state = state or {}
    intent = detect_intent(prompt)
    dbg: dict[str, Any] = {"intent": structured_intent_to_dict(intent)}

    crs = state.get("context_run_state")
    orch = state.get("orchestrator")
    if crs is None and isinstance(orch, ContextPipelineOrchestrator):
        crs = orch.state
    if crs is None or not isinstance(crs, ContextRunState):
        dbg.update({"retrieved": 0, "reranked": 0, "final_tokens": 0, "note": "no_context_run_state"})
        return {"context": "", "intent": structured_intent_to_dict(intent), "debug": dbg}

    crs.query = prompt
    runner = orch if isinstance(orch, ContextPipelineOrchestrator) else ContextPipelineOrchestrator(crs)
    trace = runner.run_through_compression(debug=debug)
    if trace:
        dbg.update(trace)
    if getattr(crs, "subagent_debug", None):
        dbg["subagent_trace"] = crs.subagent_debug
    if getattr(crs, "context_strategy", None):
        dbg["context_strategy"] = crs.context_strategy

    structured = structured_intent_to_dict(intent)
    model = (getattr(crs, "model_profile", None) or os.environ.get("TOKEN_REDUCER_MODEL", "sonnet")).strip()
    out_budget = effective_chunk_token_budget(int(structured["token_budget"]), model)
    crs.bullets = ensure_bullets_within_token_budget(crs.bullets, out_budget)
    final_tok = _final_bullet_token_estimate(crs.bullets)
    assert final_tok <= out_budget, f"final_tokens {final_tok} exceed budget {out_budget}"

    bullets_txt = "\n".join(f"- {b}" for b in crs.bullets)
    dbg["final_tokens"] = final_tok
    dbg["output_token_budget"] = out_budget
    return {
        "context": bullets_txt,
        "intent": structured_intent_to_dict(intent),
        "debug": dbg,
    }
