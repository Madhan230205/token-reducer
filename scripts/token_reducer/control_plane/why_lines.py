"""Structured why-lines for decision_trace_v2 — categorized projections only (no free-form prose).

Each line maps to AuditSpine / GateDecision / SignalSnapshot fields or a documented derivation rule.
"""

from __future__ import annotations

from ..audit_spine import AuditSpine
from .models import GateDecisionV1, IntentTask, SignalSnapshotV1, WhyLineStruct


def _disagreement_band(d: float) -> str:
    if d < 0.15:
        return "low"
    if d < 0.35:
        return "medium"
    return "high"


def build_why_lines(
    *,
    intent_task: IntentTask,
    snapshot: SignalSnapshotV1,
    gate: GateDecisionV1,
    spine: AuditSpine,
    compression_preset: str,
    retention_estimate: float,
    compression_flags: tuple[str, ...],
    retrieval_retry_done: bool,
) -> tuple[WhyLineStruct, ...]:
    """Deterministic why lines from spine + snapshot + gate + run state."""
    disc = spine.retrieval.disagreement
    d_scalar = round(float(snapshot.disagreement), 4)
    tier = spine.routing.selected_tier
    rel = spine.routing.risk_flags
    risk_s = ", ".join(rel) if rel else "none"

    lines: list[WhyLineStruct] = [
        WhyLineStruct(category="intent", value=f"task {intent_task}"),
        WhyLineStruct(
            category="routing",
            value=f"retrieval tier {tier} (risk_flags={risk_s})",
        ),
        WhyLineStruct(
            category="weak_retrieval",
            value="true" if snapshot.weak_retrieval else "false",
        ),
        WhyLineStruct(
            category="disagreement",
            value=f"{d_scalar} ({_disagreement_band(d_scalar)}) overlap={round(float(disc.fts_vector_overlap), 4)}",
        ),
        WhyLineStruct(
            category="retrieval",
            value=f"n_candidates={len(spine.retrieval.candidates)} top_k_score={snapshot.retrieval_stats.top_k_score}",
        ),
        WhyLineStruct(
            category="compression",
            value=f"preset {compression_preset} retention={round(float(retention_estimate), 4)} flags={','.join(compression_flags) or 'none'}",
        ),
        WhyLineStruct(
            category="retry",
            value="triggered" if retrieval_retry_done else "not triggered",
        ),
        WhyLineStruct(
            category="gate",
            value=(
                f"allow_vector={gate.allow_vector} allow_retry={gate.allow_retry} "
                f"cap={gate.compression_aggressiveness_cap} explain={gate.explainability_tier}"
            ),
        ),
        WhyLineStruct(
            category="reliability",
            value=(
                f"risk_level={snapshot.risk_level} confidence={snapshot.confidence} "
                f"cache_reuse_allowed={gate.cache_reuse_allowed}"
            ),
        ),
    ]
    return tuple(lines)


def validate_why_lines(lines: tuple[WhyLineStruct, ...]) -> None:
    """Assert every line uses an allowed category (tests)."""
    allowed: set[str] = {
        "routing",
        "weak_retrieval",
        "disagreement",
        "compression",
        "retry",
        "gate",
        "reliability",
        "retrieval",
        "intent",
    }
    for w in lines:
        if w.category not in allowed:
            raise ValueError(f"orphan why category: {w.category}")
