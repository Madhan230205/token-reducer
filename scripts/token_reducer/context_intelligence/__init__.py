"""Control plane: heuristic context decisions (spec docs/superpowers/specs/…context-intelligence…)."""

from .constants import DECISION_VERSION
from .heuristic import (
    build_decision_d0,
    build_decision_d1,
    infer_task_shape,
    intel_retry_allowed,
    intent_spread_normalized,
    ranking_profile_for,
    tier_intent_disagreement,
    top3_spread,
    weak_retrieval_triggers_llm,
)
from .models import ContextDecision
from .ranking import validate_ranking_weights
from .strategy_nudge import apply_task_shape_nudge, clamp_merge_cap, clamp_prune_k
from .telemetry import build_intel_debug_payload

__all__ = [
    "DECISION_VERSION",
    "ContextDecision",
    "apply_task_shape_nudge",
    "build_decision_d0",
    "build_decision_d1",
    "build_intel_debug_payload",
    "clamp_merge_cap",
    "clamp_prune_k",
    "infer_task_shape",
    "intel_retry_allowed",
    "intent_spread_normalized",
    "ranking_profile_for",
    "tier_intent_disagreement",
    "top3_spread",
    "validate_ranking_weights",
    "weak_retrieval_triggers_llm",
]
