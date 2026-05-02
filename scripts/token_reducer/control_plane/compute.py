"""Contracts for signal fusion, gating, and decision-trace emission.

v0: structurally valid placeholders — wire to AuditSpine / ContextRunState in a later phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import (
    CompressionTrace,
    DecisionTraceV1,
    GateDecisionV1,
    SignalSnapshotV1,
    TraceBundle,
)

if TYPE_CHECKING:
    from ..audit_spine import AuditSpine
    from ..context_intelligence.models import ContextDecision


def compute_signal_snapshot(
    trace_id: str,
    *,
    audit_spine: AuditSpine | None = None,
    context_decision: ContextDecision | None = None,
) -> SignalSnapshotV1:
    """Fuse spine + context intelligence + retrieval stats into a normalized snapshot.

    Parameters are accepted for forward wiring; v0 returns defaults aside from ``trace_id``.
    """
    _ = (audit_spine, context_decision)
    return SignalSnapshotV1(trace_id=trace_id)


def compute_gate_decision(
    snapshot: SignalSnapshotV1,
    *,
    policy_profile: str = "default",
) -> GateDecisionV1:
    """Derive allowed actions from ``snapshot`` (policy engine placeholder)."""
    _ = policy_profile
    return GateDecisionV1(trace_id=snapshot.trace_id)


def emit_decision_trace(
    trace_id: str,
    *,
    snapshot: SignalSnapshotV1 | None = None,
    gate: GateDecisionV1 | None = None,
    audit_spine: AuditSpine | None = None,
) -> DecisionTraceV1:
    """Emit reviewer-facing trace (projection — not a second source of truth)."""
    _ = (snapshot, gate, audit_spine)
    return DecisionTraceV1(
        trace_id=trace_id,
        path="CACHE_MISS → RETRIEVE → SELECT → COMPRESS → PACKAGE",
        compression=CompressionTrace(preset="balanced", retention_estimate=0.0, flags=()),
        confidence=0.0,
        cache_used=False,
    )


def build_trace_bundle_skeleton(
    trace_id: str,
    *,
    audit_spine: AuditSpine | None = None,
    context_decision: ContextDecision | None = None,
) -> TraceBundle:
    """Build all three siblings with v0 placeholders — same ``trace_id`` as Audit Spine."""
    snap = compute_signal_snapshot(
        trace_id,
        audit_spine=audit_spine,
        context_decision=context_decision,
    )
    gate = compute_gate_decision(snap)
    dtrace = emit_decision_trace(
        trace_id,
        snapshot=snap,
        gate=gate,
        audit_spine=audit_spine,
    )
    return TraceBundle(
        trace_id=trace_id,
        signal_snapshot=snap,
        gate_decision=gate,
        decision_trace=dtrace,
    )
