"""On-disk trace bundle export + phase-A replay validation."""

from __future__ import annotations

from pathlib import Path

from token_reducer.audit_spine import bootstrap_spine
from token_reducer.control_plane.compute import build_trace_bundle_skeleton
from token_reducer.execution_route import ExecutionRoute
from token_reducer.intent import detect_intent
from token_reducer.trace_export import write_trace_bundle_dir
from token_reducer.trace_replay import summarize_replay


def _minimal_route() -> ExecutionRoute:
    return ExecutionRoute(
        tier="simple",
        skill_id=None,
        skip_subagents=False,
        skip_lsp=False,
        retrieval_scale=1.0,
        reducer_token_threshold=2800,
        efficiency_score=1.0,
    )


def test_write_trace_bundle_and_replay_validate(tmp_path: Path) -> None:
    route = _minimal_route()
    si = detect_intent("benchmark query")
    spine = bootstrap_spine("benchmark query", si, route)
    bundle = build_trace_bundle_skeleton(spine.trace_id, audit_spine=spine)
    root = tmp_path / "traces"
    write_trace_bundle_dir(
        root,
        spine,
        bundle,
        inputs_digest={"fixture": "unit"},
    )
    out_dir = root / spine.trace_id
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "decision_trace.json").is_file()
    summary = summarize_replay(out_dir, gate_patch_json=None)
    assert summary["valid"] is True
    assert summary["errors"] == []


def test_trace_replay_gate_patch_diff(tmp_path: Path) -> None:
    route = _minimal_route()
    si = detect_intent("x")
    spine = bootstrap_spine("x", si, route)
    bundle = build_trace_bundle_skeleton(spine.trace_id, audit_spine=spine)
    root = tmp_path / "t2"
    write_trace_bundle_dir(root, spine, bundle, inputs_digest={})
    out_dir = root / spine.trace_id
    summary = summarize_replay(out_dir, gate_patch_json='{"allow_vector": false}')
    assert summary["valid"] is True
    assert summary.get("gate_diff")
    gd = summary.get("gate_diff") or {}
    assert "allow_vector" in gd
