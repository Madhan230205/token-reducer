"""Phase-A replay: load on-disk trace artifacts, validate, optional gate patch diff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .control_plane.models import (
    DecisionTraceV2,
    GateDecisionV1,
    SignalSnapshotV1,
    TraceBundle,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_gate_patch(raw: str) -> dict[str, Any]:
    """Parse ``--gate-patch`` JSON object."""
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("gate-patch must be a JSON object")
    return data


def validate_trace_dir(trace_dir: Path) -> tuple[list[str], dict[str, Any]]:
    """Return (errors, loaded_manifest)."""
    errors: list[str] = []
    root = trace_dir.expanduser().resolve()
    if not root.is_dir():
        return [f"not a directory: {root}"], {}

    mf = root / "manifest.json"
    if not mf.is_file():
        errors.append("missing manifest.json")
        return errors, {}

    try:
        manifest = _read_json(mf)
    except (OSError, json.JSONDecodeError) as e:
        errors.append(f"manifest.json: {e}")
        return errors, {}

    if manifest.get("schema_version") != "trace_manifest_v1":
        errors.append("unexpected manifest schema_version")

    tid = manifest.get("trace_id")
    if not isinstance(tid, str) or not tid:
        errors.append("manifest.trace_id missing")

    files_obj = manifest.get("files")
    if not isinstance(files_obj, dict):
        errors.append("manifest.files must be an object")
    else:
        missing = [f for f in ("audit_spine.json", "decision_trace.json") if f not in files_obj]
        for m in missing:
            errors.append(f"missing manifest.files entry for {m}")

    for name in ("audit_spine.json", "decision_trace.json", "gate_decision.json", "signal_snapshot.json"):
        p = root / name
        if name in ("audit_spine.json", "decision_trace.json") and not p.is_file():
            errors.append(f"missing {name}")

    # Schema validate siblings when present
    dt_path = root / "decision_trace.json"
    if dt_path.is_file():
        try:
            payload = _read_json(dt_path)
            DecisionTraceV2.model_validate(payload)
        except (ValidationError, ValueError, TypeError) as e:
            errors.append(f"decision_trace.json: {e}")

    gd_path = root / "gate_decision.json"
    if gd_path.is_file():
        try:
            GateDecisionV1.model_validate(_read_json(gd_path))
        except (ValidationError, ValueError, TypeError) as e:
            errors.append(f"gate_decision.json: {e}")

    ss_path = root / "signal_snapshot.json"
    if ss_path.is_file():
        try:
            SignalSnapshotV1.model_validate(_read_json(ss_path))
        except (ValidationError, ValueError, TypeError) as e:
            errors.append(f"signal_snapshot.json: {e}")

    return errors, manifest


def gate_field_diff(before: GateDecisionV1, after: GateDecisionV1) -> dict[str, dict[str, Any]]:
    """Field-wise diff for reviewer output."""
    b = before.model_dump(mode="json")
    a = after.model_dump(mode="json")
    out: dict[str, dict[str, Any]] = {}
    for k in sorted(set(b) | set(a)):
        if b.get(k) != a.get(k):
            out[k] = {"before": b.get(k), "after": a.get(k)}
    return out


def apply_gate_patch(base: GateDecisionV1, patch: dict[str, Any]) -> GateDecisionV1:
    """Shallow merge patch keys onto gate model (validated)."""
    data = base.model_dump(mode="json")
    for key, val in patch.items():
        if key in data:
            data[key] = val
    return GateDecisionV1.model_validate(data)


def summarize_replay(
    trace_dir: Path,
    *,
    gate_patch_json: str | None,
) -> dict[str, Any]:
    """Validate bundle and optionally show gate diff."""
    errs, manifest = validate_trace_dir(trace_dir)
    result: dict[str, Any] = {"valid": len(errs) == 0, "errors": errs, "manifest": manifest}
    if errs:
        return result

    gd_path = trace_dir / "gate_decision.json"
    if not gd_path.is_file():
        result["gate_diff"] = None
        result["note"] = "no gate_decision.json"
        return result

    base = GateDecisionV1.model_validate(_read_json(gd_path))
    if not gate_patch_json or not gate_patch_json.strip():
        result["gate_diff"] = None
        result["gate_current"] = base.model_dump(mode="json")
        return result

    patch = load_gate_patch(gate_patch_json)
    patched = apply_gate_patch(base, patch)
    result["gate_diff"] = gate_field_diff(base, patched)
    result["gate_patched"] = patched.model_dump(mode="json")
    return result


def trace_bundle_from_dir(trace_dir: Path) -> TraceBundle | None:
    """Best-effort load TraceBundle JSON shapes from disk (for tooling)."""
    root = trace_dir.expanduser().resolve()
    ss_path = root / "signal_snapshot.json"
    gd_path = root / "gate_decision.json"
    dt_path = root / "decision_trace.json"
    tid_path = root / "manifest.json"
    if not tid_path.is_file():
        return None
    manifest = _read_json(tid_path)
    tid = manifest.get("trace_id")
    if not isinstance(tid, str):
        return None
    snap = SignalSnapshotV1.model_validate(_read_json(ss_path)) if ss_path.is_file() else None
    gate = GateDecisionV1.model_validate(_read_json(gd_path)) if gd_path.is_file() else None
    dtraw = _read_json(dt_path) if dt_path.is_file() else None
    dtrace = DecisionTraceV2.model_validate(dtraw) if dtraw else None
    return TraceBundle(trace_id=tid, signal_snapshot=snap, gate_decision=gate, decision_trace=dtrace)
