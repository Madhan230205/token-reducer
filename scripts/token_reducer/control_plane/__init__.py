"""Control plane artifacts — SignalSnapshot, GateDecision, DecisionTrace (siblings of Audit Spine).

Spec: docs/superpowers/specs/2026-05-02-trace-bundle-control-plane-design.md
"""

from __future__ import annotations

from .compute import (
    build_trace_bundle_from_run,
    build_trace_bundle_skeleton,
    compute_gate_decision,
    compute_signal_snapshot,
    derive_reliability,
    emit_decision_trace,
    map_intent_task,
)
from .models import (
    ChunkRefV1,
    CompressionTrace,
    DecisionTraceRetrieval,
    DecisionTraceV2,
    GateDecisionV1,
    IntentTask,
    ReliabilityTrace,
    RetrievalStats,
    SignalSnapshotV1,
    TraceBundle,
    WhyLineStruct,
)

__all__ = [
    "ChunkRefV1",
    "CompressionTrace",
    "DecisionTraceRetrieval",
    "DecisionTraceV2",
    "GateDecisionV1",
    "IntentTask",
    "ReliabilityTrace",
    "RetrievalStats",
    "SignalSnapshotV1",
    "TraceBundle",
    "WhyLineStruct",
    "build_trace_bundle_from_run",
    "build_trace_bundle_skeleton",
    "compute_gate_decision",
    "compute_signal_snapshot",
    "derive_reliability",
    "emit_decision_trace",
    "map_intent_task",
]
