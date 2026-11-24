"""ContextDecision artifact (§3) — JSON-serializable."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .constants import DECISION_VERSION

Provenance = Literal["heuristic", "llm_fallback", "blended"]
RiskLevel = Literal["low", "medium", "high"]
OverrideStrength = Literal["weak", "medium", "strong"]


@dataclass
class ContextDecision:
    decision_version: str = DECISION_VERSION
    decision_id: str = ""
    fallback_cache_key: str = ""

    task_family: str = "explain_code"
    task_shape: str = "analysis"

    compression_delta: float = 0.0
    ranking_profile: str = "default"
    ranking_weights: dict[str, float] | None = None

    confidence: float = 0.5
    confidence_threshold_used: float = 0.42
    confidence_subscores: dict[str, float] = field(default_factory=dict)
    llm_confidence: float | None = None

    risk_level: RiskLevel = "medium"

    provenance: Provenance = "heuristic"
    fallback_triggered: bool = False
    fallback_reason: str | None = None

    override_strength: OverrideStrength = "weak"

    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
