"""Control plane artifacts — SignalSnapshot, GateDecision, DecisionTrace (siblings of Audit Spine).

Spec: docs/superpowers/specs/2026-05-02-trace-bundle-control-plane-design.md
"""

from __future__ import annotations

from .compute import (
    build_trace_bundle_skeleton,
    compute_gate_decision,
    compute_signal_snapshot,
    emit_decision_trace,
)
from .models import (
    CompressionTrace,
    DecisionTraceV1,
    DroppedChunk,
    GateDecisionV1,
    KeptChunk,
    RetrievalStats,
    SignalSnapshotV1,
    TraceBundle,
)

__all__ = [
    "CompressionTrace",
    "DecisionTraceV1",
    "DroppedChunk",
    "GateDecisionV1",
    "KeptChunk",
    "RetrievalStats",
    "SignalSnapshotV1",
    "TraceBundle",
    "build_trace_bundle_skeleton",
    "compute_gate_decision",
    "compute_signal_snapshot",
    "emit_decision_trace",
]
