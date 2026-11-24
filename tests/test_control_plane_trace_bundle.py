from __future__ import annotations

from token_reducer.control_plane import (
    TraceBundle,
    build_trace_bundle_skeleton,
    compute_gate_decision,
    compute_signal_snapshot,
    emit_decision_trace,
)


def test_signal_snapshot_schema() -> None:
    s = compute_signal_snapshot("tid-1")
    assert s.schema_version == "signal_snapshot_v1"
    assert s.trace_id == "tid-1"


def test_gate_derives_trace_id() -> None:
    snap = compute_signal_snapshot("tid-2")
    g = compute_gate_decision(snap)
    assert g.schema_version == "gate_decision_v1"
    assert g.trace_id == "tid-2"


def test_emit_decision_trace_shape() -> None:
    d = emit_decision_trace("tid-3")
    assert d.schema_version == "decision_trace_v2"
    assert "→" in d.path


def test_trace_bundle_json_roundtrip_keys() -> None:
    b = build_trace_bundle_skeleton("tid-4")
    assert isinstance(b, TraceBundle)
    js = b.to_jsonable()
    assert js["trace_id"] == "tid-4"
    assert js["signal_snapshot_v1"]["trace_id"] == "tid-4"
    assert js["gate_decision_v1"]["trace_id"] == "tid-4"
    assert js["decision_trace_v2"]["trace_id"] == "tid-4"
