"""Heuristic D₀ / D₁ decision builders (spec §2, §4)."""

from __future__ import annotations

import json
import re
from hashlib import blake2b
from typing import Any

from ..intent import IntentType
from .constants import (
    DECISION_VERSION,
    DEFAULT_CONFIDENCE_THRESHOLD,
    TOP3_SPREAD_RETRY_THRESHOLD,
    WEAK_RETRIEVAL_CONFIDENCE_MARGIN,
)
from .models import ContextDecision, RiskLevel


def _stable_hash(parts: list[str]) -> str:
    h = blake2b(digest_size=16)
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def infer_task_shape(query: str, legacy_intent: str) -> str:
    q = (query or "").strip().lower()
    if re.search(r"\b(vs\.?|versus|compare|difference between|diff\b)\b", q):
        return "comparison"
    if re.search(
        r"\b(implement|write\s+(the\s+)?code|scaffold|generate|create\s+(a\s+)?(new\s+)?(class|function|api))\b",
        q,
    ):
        return "generation"
    if re.search(r"\b(step\s+by\s+step|iterate|refinement|incrementally)\b", q):
        return "iterative_refinement"
    wc = len(q.split())
    if wc > 40 or legacy_intent == "explain_code":
        return "analysis"
    if wc < 12:
        return "single_shot"
    return "analysis"


def ranking_profile_for(shape: str, legacy: str) -> str:
    if shape == "comparison" or legacy == "navigation":
        return "lexical_heavy"
    if shape == "generation":
        return "semantic_heavy"
    return "default"


def intent_spread_normalized(scores: dict[IntentType, int]) -> float:
    vals = sorted((int(v) for v in scores.values()), reverse=True)
    if len(vals) < 2:
        return 0.0
    top, second = vals[0], vals[1]
    if top <= 0:
        return 0.0
    return max(0.0, min(1.0, (top - second) / float(top + second + 1)))


def top3_spread(pool: list[Any]) -> float:
    if len(pool) < 3:
        return 0.0
    ranked = sorted(pool, key=lambda c: float(c.final_score), reverse=True)
    s0, s2 = float(ranked[0].final_score), float(ranked[2].final_score)
    return max(0.0, s0 - s2)


def tier_intent_disagreement(tier: str | None, legacy: str) -> float:
    """Heuristic mismatch signal ∈ [0,1]."""
    if not tier:
        return 0.0
    if tier == "simple" and legacy in ("bug_fix", "feature_add"):
        return 0.45
    if tier == "complex" and legacy == "explain_code":
        return 0.15
    return 0.0


def build_decision_d0(
    query: str,
    structured: dict[str, Any],
    *,
    execution_tier: str | None,
    confidence_threshold: float | None = None,
) -> ContextDecision:
    legacy = str(structured.get("legacy_intent", "explain_code"))
    shape = infer_task_shape(query, legacy)
    profile = ranking_profile_for(shape, legacy)
    thr = float(confidence_threshold if confidence_threshold is not None else DEFAULT_CONFIDENCE_THRESHOLD)
    td = tier_intent_disagreement(execution_tier, legacy)
    risk: RiskLevel = "medium" if td > 0.3 else "low"
    return ContextDecision(
        task_family=legacy,
        task_shape=shape,
        compression_delta=0.0,
        ranking_profile=profile,
        ranking_weights=None,
        confidence=0.55,
        confidence_threshold_used=thr,
        risk_level=risk,
        provenance="heuristic",
        signals={"phase": "d0", "execution_tier": execution_tier, "tier_disagreement": td},
    )


def build_decision_d1(
    d0: ContextDecision,
    *,
    query: str,
    scores: dict[IntentType, int],
    fts_hits: int,
    vector_hits: int,
    min_fts_hits: int,
    scored_pool: list[Any],
    weak_pool_legacy: bool,
) -> ContextDecision:
    spread_i = intent_spread_normalized(scores)
    t3 = top3_spread(scored_pool)
    weak = bool(fts_hits < min_fts_hits or weak_pool_legacy)

    disagreement = float(d0.signals.get("tier_disagreement", 0.0))
    et = d0.signals.get("execution_tier")
    if et is not None:
        disagreement = max(disagreement, tier_intent_disagreement(str(et), d0.task_family))

    conf = (
        0.38
        + 0.28 * spread_i
        + 0.18 * min(1.0, t3 * 4.0)
        - (0.22 if weak else 0.0)
        - 0.24 * disagreement
    )
    conf = max(0.05, min(0.95, conf))

    risk: RiskLevel
    if weak and conf < 0.45:
        risk = "high"
    elif weak or disagreement > 0.25 or conf < 0.38:
        risk = "medium"
    else:
        risk = "low"

    comp_delta = 0.0
    if risk == "high":
        comp_delta = -0.15
    elif risk == "medium":
        comp_delta = -0.08

    sig_summary = json.dumps(
        {"fts": fts_hits, "vec": vector_hits, "weak": weak, "t3": round(t3, 4), "spr": round(spread_i, 4)},
        sort_keys=True,
    )
    qn = " ".join((query or "").strip().lower().split())
    fb_key = _stable_hash([qn, sig_summary, DECISION_VERSION])
    did = _stable_hash([fb_key, DECISION_VERSION])

    return ContextDecision(
        decision_version=DECISION_VERSION,
        decision_id=did,
        fallback_cache_key=fb_key,
        task_family=d0.task_family,
        task_shape=d0.task_shape,
        compression_delta=comp_delta,
        ranking_profile=d0.ranking_profile,
        ranking_weights=d0.ranking_weights,
        confidence=conf,
        confidence_threshold_used=d0.confidence_threshold_used,
        confidence_subscores={
            "intent_spread": spread_i,
            "disagreement": disagreement,
            "top3_spread": t3,
        },
        risk_level=risk,
        provenance="heuristic",
        signals={
            "phase": "d1",
            "intent_spread": spread_i,
            "disagreement": disagreement,
            "weak_retrieval": weak,
            "top3_spread": t3,
            "fts_hits": fts_hits,
            "vector_hits": vector_hits,
            "execution_tier": d0.signals.get("execution_tier"),
        },
    )


def intel_retry_allowed(decision: ContextDecision, weak_pool_legacy: bool) -> bool:
    """§5.5 — high risk AND (weak_retrieval OR low top-3 spread)."""
    if decision.risk_level != "high":
        return False
    weak = bool(decision.signals.get("weak_retrieval") or weak_pool_legacy)
    t3 = float(decision.signals.get("top3_spread") or 0.0)
    low_spread = t3 < TOP3_SPREAD_RETRY_THRESHOLD
    return weak or low_spread


def weak_retrieval_triggers_llm(confidence: float, threshold: float, weak: bool) -> bool:
    return weak and confidence < (threshold + WEAK_RETRIEVAL_CONFIDENCE_MARGIN)
