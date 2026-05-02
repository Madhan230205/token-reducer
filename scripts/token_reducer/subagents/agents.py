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


def _legacy_intent(intent: dict[str, Any]) -> str:
    return str(_intent_dict(intent).get("legacy_intent", "explain_code"))


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
        _ = prompt
        if not chunks:
            return chunks
        leg = _legacy_intent(state.get("structured_intent") or {})
        top = max(float(c.final_score) for c in chunks)
        # Precision for navigation; recall-ish for bug_fix; balanced otherwise
        peak_mult = {"navigation": 0.44, "bug_fix": 0.31, "explain_code": 0.35, "feature_add": 0.34, "refactor": 0.34}.get(
            leg, 0.36
        )
        thr = max(0.055, peak_mult * top)
        filt = [c for c in chunks if float(c.final_score) >= thr]
        if not filt:
            filt = chunks[:3]
        dedupe = {"navigation": 0.93, "bug_fix": 0.86}.get(leg, 0.9)
        return remove_redundant_chunks(filt, threshold=dedupe)


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
        skill_nudge = float(state.get("adaptive_skill_prior_nudge") or 0.0)
        paths: list[str] = (state.get("run_memory") or {}).get("focus_paths") or []
        path_bump = 0.022
        for c in chunks:
            fs = float(c.final_score) + w + skill_nudge
            fs += float(boosts.get(Path(c.source).name, 0.0))
            if paths:
                cnorm = str(c.source).replace("\\", "/").lower()
                if any(p.replace("\\", "/").lower() in cnorm or Path(p).name.lower() in cnorm for p in paths):
                    fs += path_bump + (0.014 if legacy == "navigation" else 0.0)
            terms: list[str] = (state.get("run_memory") or {}).get("focus_terms") or []
            if terms:
                blob = (c.text + " " + c.source).lower()
                hits = sum(1 for t in terms if t.lower() in blob)
                if hits:
                    fs += 0.012 * min(hits, 4)
            if legacy == "bug_fix":
                blob2 = (c.text + c.source).lower()
                if any(k in blob2 for k in ("raise ", "except", "traceback", "error", "assert ", "panic")):
                    fs += 0.042
            c.final_score = fs
        return sorted(chunks, key=lambda x: x.final_score, reverse=True)


class VarianceAgent(SubAgent):
    """Deterministic nudge: occasionally surface a strong chunk from a less obvious file."""

    name = "variance"

    def should_run(self, intent: dict[str, Any], state: dict[str, Any]) -> bool:
        n = int(state.get("_chunk_count", 0))
        leg = _legacy_intent(intent)
        if leg == "navigation":
            return False
        if leg == "explain_code":
            return n >= 5
        if leg == "bug_fix":
            return n >= 8
        _ = intent
        return n >= 7

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
        n = int(state.get("_chunk_count", 0))
        leg = _legacy_intent(intent)
        hi = 36 if leg == "bug_fix" else 28
        return 2 <= n <= hi

    def run(self, chunks: list[Candidate], prompt: str, state: dict[str, Any]) -> list[Candidate]:
        _ = prompt
        if not chunks:
            return chunks
        leg = _legacy_intent(state.get("structured_intent") or {})
        sim_need = 0.68 if leg == "bug_fix" else 0.72
        fused: list[Candidate] = []
        buffer: Candidate | None = None
        for c in chunks:
            if buffer is None:
                buffer = c
                continue
            same = buffer.source == c.source
            sim = jaccard_text(buffer.text, c.text)
            if same and sim > sim_need:
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
