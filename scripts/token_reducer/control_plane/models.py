"""Typed v1 artifacts for TraceBundle (policy siblings — not embedded in Audit Spine)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SignalSnapshotSchema = Literal["signal_snapshot_v1"]
GateDecisionSchema = Literal["gate_decision_v1"]
DecisionTraceSchema = Literal["decision_trace_v1"]

RiskLevel = Literal["low", "medium", "high"]
CompressionAggCap = Literal["low", "medium", "high"]
ExplainabilityTier = Literal["minimal", "standard", "full"]


class RetrievalStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    top_k_score: float = 0.0
    coverage: float = 0.0


class SignalSnapshotV1(BaseModel):
    """Normalized inputs to the policy layer (v1)."""

    model_config = ConfigDict(frozen=True)

    schema_version: SignalSnapshotSchema = "signal_snapshot_v1"
    trace_id: str
    confidence: float = 0.0
    risk_level: RiskLevel = "medium"
    disagreement: float = 0.0
    weak_retrieval: bool = False
    retrieval_stats: RetrievalStats = Field(default_factory=RetrievalStats)

    def to_jsonable(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class GateDecisionV1(BaseModel):
    """Policy interpretation — what the run was allowed to do (v1)."""

    model_config = ConfigDict(frozen=True)

    schema_version: GateDecisionSchema = "gate_decision_v1"
    trace_id: str
    allow_vector: bool = True
    allow_retry: bool = True
    compression_aggressiveness_cap: CompressionAggCap = "medium"
    cache_reuse_allowed: bool = False
    explainability_tier: ExplainabilityTier = "full"

    def to_jsonable(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class KeptChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    reason: str


class DroppedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    reason: str


class CompressionTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    preset: str = "balanced"
    retention_estimate: float = 0.0
    flags: tuple[str, ...] = ()


class DecisionTraceV1(BaseModel):
    """Reviewer-facing outcome summary (v1); curated projection of the run."""

    model_config = ConfigDict(frozen=True)

    schema_version: DecisionTraceSchema = "decision_trace_v1"
    trace_id: str
    path: str = ""
    kept: tuple[KeptChunk, ...] = ()
    dropped: tuple[DroppedChunk, ...] = ()
    compression: CompressionTrace = Field(default_factory=CompressionTrace)
    confidence: float = 0.0
    cache_used: bool = False

    def to_jsonable(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class TraceBundle(BaseModel):
    """Sibling bundle keyed by ``trace_id`` (same id as Audit Spine)."""

    model_config = ConfigDict(frozen=True)

    trace_id: str
    signal_snapshot: SignalSnapshotV1 | None = None
    gate_decision: GateDecisionV1 | None = None
    decision_trace: DecisionTraceV1 | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "signal_snapshot_v1": self.signal_snapshot.to_jsonable()
            if self.signal_snapshot
            else None,
            "gate_decision_v1": self.gate_decision.to_jsonable() if self.gate_decision else None,
            "decision_trace_v1": self.decision_trace.to_jsonable() if self.decision_trace else None,
        }
