from __future__ import annotations

from token_reducer.audit_spine import AuditSpine
from token_reducer.explainability_render import (
    attach_hybrid_explainability,
    confidence_band,
    render_payload_one_liner,
    render_sidecar_text,
)


def _minimal_spine_dict() -> dict:
    return {
        "schema_version": "audit_spine_v0_1",
        "trace_id": "00000000-0000-4000-8000-000000000001",
        "timestamp": "2026-05-02T00:00:00+00:00",
        "query": {"raw": "", "normalized": "", "features": {"length": 3, "type": "complex"}},
        "retrieval": {
            "candidates": [
                {
                    "doc_id": "x",
                    "path": "/repo/auth.py",
                    "chunk_id": "10",
                    "source": "fts",
                    "score": 0.9,
                    "rank": 1,
                },
                {
                    "doc_id": "y",
                    "path": "/repo/session.ts",
                    "chunk_id": "11",
                    "source": "hybrid",
                    "score": 0.8,
                    "rank": 2,
                },
            ],
            "disagreement": {
                "fts_vector_overlap": 1.0,
                "rank_correlation": 1.0,
                "low_overlap_flag": False,
                "mismatch_flag": False,
            },
        },
        "routing": {
            "selected_tier": "complex",
            "thresholds": {"confidence": 0.5, "risk": 0.5},
            "risk_flags": [],
            "decision_reason": "",
        },
        "compression": {
            "input_tokens": 100,
            "output_tokens": 80,
            "retained_chunks": ("10", "11"),
            "dropped_chunks": ("99",),
            "loss_metrics": {"importance_retained_ratio": 0.8, "salient_span_checksum": "abc"},
        },
        "outcome": {"latency_ms": 50, "final_confidence": 0.78},
        "adaptive": {"signals": {"retrieval_usefulness": None, "compression_loss": None}, "updates": []},
    }


def test_confidence_band_stable() -> None:
    spine = _minimal_spine_dict()
    assert confidence_band(spine) == "high"


def test_one_liner_byte_stable() -> None:
    spine = _minimal_spine_dict()
    line = render_payload_one_liner(spine, preset="balanced")
    assert line.startswith("Context optimized (balanced, high confidence). Key focus:")
    assert "auth.py" in line or "session.ts" in line
    assert len(line) <= 180


def test_sidecar_compact_within_six_lines() -> None:
    spine = _minimal_spine_dict()
    text = render_sidecar_text(
        spine,
        preset="balanced",
        patch_first=False,
        focus_paths=(),
    )
    assert text.count("\n") + 1 <= 6
    assert "Selected:" in text
    assert "Dropped:" in text
    assert "Compression:" in text


def test_attach_sets_keys() -> None:
    spine = AuditSpine.model_validate(_minimal_spine_dict())
    payload: dict[str, object] = {"summary": "x", "intent": "explain_code"}
    attach_hybrid_explainability(
        payload,
        spine,
        subagent_profile_used="full",
        patch_first=False,
    )
    assert isinstance(payload.get("context_optimization_line"), str)
    ce = payload.get("context_explainability")
    assert isinstance(ce, dict)
    assert ce.get("schema_version") == "explain_hybrid_v1"
    assert isinstance(ce.get("sidecar"), str)
    assert ce.get("verbose") is None
