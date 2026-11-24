"""Why-line categories are closed-set (no orphan strings)."""

from __future__ import annotations

from token_reducer.audit_spine import AuditSpine, new_audit_spine
from token_reducer.control_plane.compute import compute_gate_decision, compute_signal_snapshot
from token_reducer.control_plane.models import GateDecisionV1, SignalSnapshotV1
from token_reducer.control_plane.why_lines import build_why_lines, validate_why_lines


def test_build_why_lines_validate_all_categories() -> None:
    spine, _, _ = new_audit_spine()
    snap = SignalSnapshotV1(trace_id=spine.trace_id, confidence=0.7)
    gate = GateDecisionV1(trace_id=spine.trace_id)
    lines = build_why_lines(
        intent_task="general",
        snapshot=snap,
        gate=gate,
        spine=spine,
        compression_preset="balanced",
        retention_estimate=0.5,
        compression_flags=("weak_retrieval",),
        retrieval_retry_done=False,
    )
    validate_why_lines(lines)
    assert lines[0].category == "intent"


def test_why_lines_include_intent_and_routing() -> None:
    spine, _, _ = new_audit_spine()
    snap = compute_signal_snapshot(spine.trace_id, audit_spine=spine)
    gate = compute_gate_decision(snap)
    assert isinstance(spine, AuditSpine)
    lines = build_why_lines(
        intent_task="semantic_code_search",
        snapshot=snap,
        gate=gate,
        spine=spine,
        compression_preset="balanced",
        retention_estimate=0.0,
        compression_flags=(),
        retrieval_retry_done=True,
    )
    cats = {w.category for w in lines}
    assert cats == {
        "intent",
        "routing",
        "weak_retrieval",
        "disagreement",
        "retrieval",
        "compression",
        "retry",
        "gate",
        "reliability",
    }
