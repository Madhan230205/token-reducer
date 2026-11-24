"""§7 telemetry payload for context intelligence."""

from __future__ import annotations

from typing import Any

from .models import ContextDecision


def build_intel_debug_payload(
    decision: ContextDecision,
    *,
    cache_hit: bool = False,
    key_prefix: str = "",
) -> dict[str, Any]:
    return {
        "decision_id": decision.decision_id,
        "decision_version": decision.decision_version,
        "phase": str(decision.signals.get("phase", "d1")),
        "provenance": decision.provenance,
        "fallback_triggered": decision.fallback_triggered,
        "fallback_reason": decision.fallback_reason,
        "confidence": round(decision.confidence, 4),
        "confidence_threshold_used": decision.confidence_threshold_used,
        "risk_level": decision.risk_level,
        "override_strength": decision.override_strength,
        "signals": {
            "intent_spread": round(float(decision.signals.get("intent_spread", 0.0)), 4),
            "disagreement": round(float(decision.signals.get("disagreement", 0.0)), 4),
            "weak_retrieval": bool(decision.signals.get("weak_retrieval", False)),
            "top3_spread": round(float(decision.signals.get("top3_spread", 0.0)), 4),
            "fts_hits": int(decision.signals.get("fts_hits", 0)),
            "vector_hits": int(decision.signals.get("vector_hits", 0)),
        },
        "cache": {"hit": cache_hit, "fallback_cache_key_prefix": key_prefix[:16]},
    }
