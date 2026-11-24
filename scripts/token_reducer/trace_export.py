"""Write TraceBundle + Audit Spine to a stable on-disk directory layout."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .audit_spine import AuditSpine
from .control_plane.models import TraceBundle

TRACE_MANIFEST_SCHEMA = "trace_manifest_v1"

# JSON: sorted keys, 2-space indent, trailing newline — reproducible byte hashes.


def stable_json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_trace_summary(bundle: TraceBundle, spine: AuditSpine) -> str:
    """Deterministic 8–12 line human summary (no LLM)."""
    dt = bundle.decision_trace
    snap = bundle.signal_snapshot
    lines: list[str] = []
    lines.append(f"trace_id: {spine.trace_id}")
    lines.append(f"path: {dt.path if dt else '(missing)'}")
    if dt:
        lines.append(f"confidence: {dt.confidence} | routing_tier: {dt.routing_tier} | intent_task: {dt.intent_task}")
        if dt.strategy_id:
            lines.append(f"strategy_id: {dt.strategy_id}")
        rel = dt.reliability
        lines.append(
            f"reliability: low_confidence={rel.low_confidence} "
            f"weak_retrieval={rel.weak_retrieval} high_disagreement={rel.high_disagreement}"
        )
        lines.append(
            f"headline: disagreement={dt.headline_disagreement} weak_retrieval={dt.headline_weak_retrieval} "
            f"risk={dt.headline_risk_level} gate_cap={dt.headline_gate_compression_cap} retry={dt.retrieval_retry_triggered}"
        )
        retr = dt.retrieval
        lines.append(
            f"retrieval: top_score={retr.top_score} n_candidates={retr.n_candidates} "
            f"fts_vector_overlap={retr.fts_vector_overlap}"
        )
        lines.append(f"compression: {dt.compression.preset} retention={dt.compression.retention_estimate}")
        if dt.compression.flags:
            lines.append(f"compression_flags: {', '.join(dt.compression.flags)}")
        lines.append(
            f"selected_chunks: {len(dt.selected_chunks)} | dropped_chunks: {len(dt.dropped_chunks)}"
        )
        lines.append(f"cache_used: {dt.cache_used}")
        if dt.why:
            lines.append("why_display:")
            for w in dt.why:
                lines.append(f"  - {w.category}: {w.value}")
    if snap:
        lines.append(
            f"snapshot: risk={snap.risk_level} weak_retrieval={snap.weak_retrieval} "
            f"disagreement={snap.disagreement}"
        )
    if bundle.gate_decision:
        g = bundle.gate_decision
        lines.append(
            f"gate: allow_vector={g.allow_vector} allow_retry={g.allow_retry} "
            f"cap={g.compression_aggressiveness_cap} explain={g.explainability_tier}"
        )
    lines.append(f"schema: audit_spine={spine.schema_version}")
    return "\n".join(lines) + "\n"


def write_trace_bundle_dir(
    trace_root: Path,
    spine: AuditSpine,
    bundle: TraceBundle,
    *,
    inputs_digest: dict[str, Any],
) -> Path:
    """Write ``<trace_root>/<trace_id>/`` with manifest, JSON siblings, summary.

    Returns the per-run directory path.
    """
    tid = spine.trace_id
    out = trace_root.expanduser().resolve() / tid
    out.mkdir(parents=True, exist_ok=True)

    spine_payload = spine.to_jsonable()

    files_written: dict[str, str] = {}

    def _write(name: str, payload: dict[str, Any]) -> None:
        raw = stable_json_bytes(payload)
        p = out / name
        p.write_bytes(raw)
        files_written[name] = _sha256_bytes(raw)

    _write("audit_spine.json", spine_payload)
    if bundle.signal_snapshot:
        _write("signal_snapshot.json", bundle.signal_snapshot.to_jsonable())
    if bundle.gate_decision:
        _write("gate_decision.json", bundle.gate_decision.to_jsonable())
    if bundle.decision_trace:
        _write("decision_trace.json", bundle.decision_trace.to_jsonable())

    summary_text = render_trace_summary(bundle, spine)
    summary_bytes = summary_text.encode("utf-8")
    sp = out / "summary.txt"
    sp.write_bytes(summary_bytes)
    files_written["summary.txt"] = _sha256_bytes(summary_bytes)

    manifest: dict[str, Any] = {
        "schema_version": TRACE_MANIFEST_SCHEMA,
        "trace_id": tid,
        "created_at": datetime.now(UTC).isoformat(),
        "audit_spine_schema": spine.schema_version,
        "inputs_digest": inputs_digest,
        "files": {k: f"sha256:{v}" for k, v in sorted(files_written.items())},
    }
    manifest_bytes = stable_json_bytes(manifest)
    mp = out / "manifest.json"
    mp.write_bytes(manifest_bytes)
    return out
