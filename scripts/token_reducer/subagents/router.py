"""Single-pass subagent runner (pre-compression, deterministic)."""

from __future__ import annotations

import os
from typing import Any

from ..intent import structured_intent_to_dict
from ..models import Candidate
from . import agents as _agents_bootstrap  # noqa: F401
from .registry import get_agents

# Upper bound; per-run slice comes from autonomous strategy (``max_chunks`` in router state).
MAX_SUBAGENT_INPUT = 18


def run_subagents(
    chunks: list[Candidate],
    prompt: str,
    intent: Any,
    state: dict[str, Any],
) -> tuple[list[Candidate], dict[str, Any]]:
    """Run coordinator → merge_streams → filter → ranking → variance → fusion → budget; return debug metadata."""
    _ = _agents_bootstrap
    if not chunks:
        return [], {
            "subagents": [],
            "chunks_before": 0,
            "chunks_after": 0,
            "tokens_before": 0,
            "tokens_after": 0,
            "tokens_saved": 0,
        }

    ranked = sorted(chunks, key=lambda c: float(c.final_score), reverse=True)
    cap = int(state.get("max_chunks") or MAX_SUBAGENT_INPUT)
    cap = max(4, min(MAX_SUBAGENT_INPUT, cap))
    work = ranked[:cap]
    before_n = len(work)
    before_tok = sum(int(c.token_estimate) for c in work)

    sid: dict[str, Any]
    if isinstance(intent, dict):
        sid = dict(intent)
    else:
        sid = structured_intent_to_dict(intent)
    run_state: dict[str, Any] = {
        **state,
        "structured_intent": sid,
        "_chunk_count": before_n,
        "model": state.get("model") or os.environ.get("TOKEN_REDUCER_MODEL", "sonnet"),
        "skip_fusion": bool(state.get("skip_fusion", False)),
    }

    ran: list[str] = []
    cur = work
    for agent in get_agents():
        run_state["_chunk_count"] = len(cur)
        if agent.should_run(sid, run_state):
            cur = agent.run(cur, prompt, run_state)
            ran.append(agent.name)

    after_n = len(cur)
    after_tok = sum(int(c.token_estimate) for c in cur)

    meta = {
        "subagents": ran,
        "chunks_before": before_n,
        "chunks_after": after_n,
        "tokens_before": before_tok,
        "tokens_after": after_tok,
        "tokens_saved": max(0, before_tok - after_tok),
    }
    rm = state.get("run_memory")
    if isinstance(rm, dict) and rm:
        meta["run_memory"] = {
            "decomposition": list(rm.get("decomposition") or []),
            "focus_paths": list(rm.get("focus_paths") or []),
            "focus_terms": list(rm.get("focus_terms") or []),
        }
    return cur, meta
