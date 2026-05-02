"""Deterministic subagents: coordinator → merge_streams → filter → ranking → variance → fusion → budget."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..compressor import remove_redundant_chunks
from ..models import Candidate
from .base import SubAgent
from .coordination import build_run_memory, diversify_chunk_order, merge_subtask_streams
from .policy import effective_chunk_token_budget
from .registry import register
from .similarity import jaccard_text


def _intent_dict(intent: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(intent, dict):
        return intent
    return dict(intent)


class CoordinatorAgent(SubAgent):
    """Fills shared ``run_memory`` (decomposition + focus signals) for the same-pass agents."""

    name = "coordinator"

    def should_run(self, intent: dict[str, Any], state: dict[str, Any]) -> bool:
        _ = state
        return bool(_intent_dict(intent))

    def run(self, chunks: list[Candidate], prompt: str, state: dict[str, Any]) -> list[Candidate]:
        sid = _intent_dict(state.get("structured_intent") or {})
        state.setdefault("run_memory", {})
        state["run_memory"].update(build_run_memory(prompt, sid))
        return chunks


class MergeStreamsAgent(SubAgent):
    """Path stream + term stream + remainder → one merged list (assign / merge, no LLM)."""

    name = "merge_streams"

    def should_run(self, intent: dict[str, Any], state: dict[str, Any]) -> bool:
        _ = intent
        if int(state.get("_chunk_count", 0)) <= 2:
            return False
        rm = state.get("run_memory") or {}
        return bool(rm.get("focus_paths")) or bool(rm.get("focus_terms"))

    def run(self, chunks: list[Candidate], prompt: str, state: dict[str, Any]) -> list[Candidate]:
        _ = prompt
        rm = state.get("run_memory") or {}
        return merge_subtask_streams(chunks, rm if isinstance(rm, dict) else {})


class FilterAgent(SubAgent):
    """Relevance gate + near-duplicate removal (single pass)."""

    name = "filter"

    def should_run(self, intent: dict[str, Any], state: dict[str, Any]) -> bool:
        _ = intent
        return int(state.get("_chunk_count", 0)) > 1

    def run(self, chunks: list[Candidate], prompt: str, state: dict[str, Any]) -> list[Candidate]:
        _ = state
        if not chunks:
            return chunks
        top = max(float(c.final_score) for c in chunks)
        thr = max(0.07, 0.36 * top)
        filt = [c for c in chunks if float(c.final_score) >= thr]
        if not filt:
            filt = chunks[:3]
        return remove_redundant_chunks(filt, threshold=0.9)


class RankingAgent(SubAgent):
    """Intent + feedback boosts, then resort."""

    name = "ranking"

    def should_run(self, intent: dict[str, Any], state: dict[str, Any]) -> bool:
        _ = state
        d = _intent_dict(intent)
        return d.get("type") in ("code", "analysis", "chat")

    def run(self, chunks: list[Candidate], prompt: str, state: dict[str, Any]) -> list[Candidate]:
        _ = prompt
        d = _intent_dict(state.get("structured_intent") or {})
        legacy = str(d.get("legacy_intent", "explain_code"))
        bump: dict[str, float] = {
            "bug_fix": 0.06,
            "feature_add": 0.045,
            "refactor": 0.035,
            "navigation": 0.05,
            "explain_code": 0.02,
        }
        w = bump.get(legacy, 0.02)
        boosts: dict[str, float] = state.get("feedback_source_boost") or {}
        paths: list[str] = (state.get("run_memory") or {}).get("focus_paths") or []
        path_bump = 0.022
        for c in chunks:
            fs = float(c.final_score) + w
            fs += float(boosts.get(Path(c.source).name, 0.0))
            if paths:
                cnorm = str(c.source).replace("\\", "/").lower()
                if any(p.replace("\\", "/").lower() in cnorm or Path(p).name.lower() in cnorm for p in paths):
                    fs += path_bump
            terms: list[str] = (state.get("run_memory") or {}).get("focus_terms") or []
            if terms:
                blob = (c.text + " " + c.source).lower()
                hits = sum(1 for t in terms if t.lower() in blob)
                if hits:
                    fs += 0.012 * min(hits, 4)
            c.final_score = fs
        return sorted(chunks, key=lambda x: x.final_score, reverse=True)


class VarianceAgent(SubAgent):
    """Deterministic nudge: occasionally surface a strong chunk from a less obvious file."""

    name = "variance"

    def should_run(self, intent: dict[str, Any], state: dict[str, Any]) -> bool:
        _ = intent
        return int(state.get("_chunk_count", 0)) >= 7

    def run(self, chunks: list[Candidate], prompt: str, state: dict[str, Any]) -> list[Candidate]:
        _ = state
        return diversify_chunk_order(chunks, prompt)


def _fuse_pair(left: Candidate, right: Candidate) -> Candidate:
    text = left.text + "\n" + right.text
    te = int(left.token_estimate) + int(right.token_estimate)
    fs = max(float(left.final_score), float(right.final_score))
    return left.model_copy(
        update={
            "text": text,
            "token_estimate": max(1, te),
            "final_score": fs,
        }
    )


class FusionAgent(SubAgent):
    name = "fusion"

    def should_run(self, intent: dict[str, Any], state: dict[str, Any]) -> bool:
        if state.get("skip_fusion"):
            return False
        _ = intent
        n = int(state.get("_chunk_count", 0))
        return 2 <= n <= 32

    def run(self, chunks: list[Candidate], prompt: str, state: dict[str, Any]) -> list[Candidate]:
        _ = prompt, state
        if not chunks:
            return chunks
        fused: list[Candidate] = []
        buffer: Candidate | None = None
        for c in chunks:
            if buffer is None:
                buffer = c
                continue
            same = buffer.source == c.source
            sim = jaccard_text(buffer.text, c.text)
            if same and sim > 0.72:
                buffer = _fuse_pair(buffer, c)
            else:
                fused.append(buffer)
                buffer = c
        if buffer is not None:
            fused.append(buffer)
        return fused


class BudgetAgent(SubAgent):
    name = "budget"

    def should_run(self, intent: dict[str, Any], state: dict[str, Any]) -> bool:
        _ = intent, state
        return True

    def run(self, chunks: list[Candidate], prompt: str, state: dict[str, Any]) -> list[Candidate]:
        _ = prompt
        d = _intent_dict(state.get("structured_intent") or {})
        base_tb = int(d.get("token_budget", 2000))
        model = state.get("model")
        budget = effective_chunk_token_budget(base_tb, model)
        result: list[Candidate] = []
        total = 0
        for c in chunks:
            te = int(c.token_estimate)
            if total + te <= budget:
                result.append(c)
                total += te
            else:
                break
        if result:
            return result
        if not chunks:
            return []
        under = [c for c in chunks if int(c.token_estimate) <= budget]
        if under:
            return [max(under, key=lambda c: float(c.final_score))]
        return [max(chunks, key=lambda c: float(c.final_score))]


def _register_all() -> None:
    from . import registry as reg

    if reg.SUBAGENTS and getattr(reg.SUBAGENTS[0], "name", "") == "coordinator":
        return
    reg.SUBAGENTS.clear()
    register(CoordinatorAgent())
    register(MergeStreamsAgent())
    register(FilterAgent())
    register(RankingAgent())
    register(VarianceAgent())
    register(FusionAgent())
    register(BudgetAgent())


_register_all()
