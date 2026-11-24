"""Typed v1/v2 artifacts for TraceBundle (policy siblings — not embedded in Audit Spine)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SignalSnapshotSchema = Literal["signal_snapshot_v1"]
GateDecisionSchema = Literal["gate_decision_v1"]
DecisionTraceSchema = Literal["decision_trace_v2"]

RiskLevel = Literal["low", "medium", "high"]
CompressionAggCap = Literal["low", "medium", "high"]
ExplainabilityTier = Literal["minimal", "standard", "full"]

IntentTask = Literal[
    "semantic_code_search",
    "qa_lookup",
    "tool_execution",
    "multi_hop_reasoning",
    "general",
    "unknown",
]


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


class ChunkRefV1(BaseModel):
    """Selected or dropped chunk row (id + path + reason)."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    path: str
    reason: str


class CompressionTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    preset: str = "balanced"
    retention_estimate: float = 0.0
    flags: tuple[str, ...] = ()


class ReliabilityTrace(BaseModel):
    """Deterministic trust flags projected from spine + snapshot (v1)."""

    model_config = ConfigDict(frozen=True)

    low_confidence: bool = False
    weak_retrieval: bool = False
    high_disagreement: bool = False


class DecisionTraceRetrieval(BaseModel):
    """Retrieval strength summary — strict projection from AuditSpine retrieval slice."""

    model_config = ConfigDict(frozen=True)

    top_score: float = 0.0
    n_candidates: int = 0
    fts_vector_overlap: float = 1.0


ExecutionTierStr = Literal["simple", "tool", "complex"]


class WhyLineStruct(BaseModel):
    """Structured why line (canonical JSON); display as ``category: value``."""

    model_config = ConfigDict(frozen=True)

    category: str
    value: str


class DecisionTraceV2(BaseModel):
    """Reviewer-facing outcome summary (v2); self-contained curated projection of the run."""

    model_config = ConfigDict(frozen=True)

    schema_version: DecisionTraceSchema = "decision_trace_v2"
    trace_id: str

    intent_task: IntentTask = "unknown"
    strategy_id: str | None = None

    tier: ExecutionTierStr = "simple"
    routing_tier: ExecutionTierStr = "simple"

    path: str = ""
    selected_chunks: tuple[ChunkRefV1, ...] = ()
    dropped_chunks: tuple[ChunkRefV1, ...] = ()

    why: tuple[WhyLineStruct, ...] = ()

    compression: CompressionTrace = Field(default_factory=CompressionTrace)
    confidence: float = 0.0
    cache_used: bool = False
    reliability: ReliabilityTrace = Field(default_factory=ReliabilityTrace)
    retrieval: DecisionTraceRetrieval = Field(default_factory=DecisionTraceRetrieval)

    headline_disagreement: float = 0.0
    headline_weak_retrieval: bool = False
    headline_risk_level: RiskLevel = "medium"
    headline_explainability_tier: ExplainabilityTier = "full"
    headline_gate_allow_vector: bool = True
    headline_gate_allow_retry: bool = True
    headline_gate_compression_cap: CompressionAggCap = "medium"
    retrieval_retry_triggered: bool = False

    tool_skill_id: str | None = None
    subagents_used: tuple[str, ...] = ()
    compression_triggered: bool = False

    def to_jsonable(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["why_display"] = [f"{w['category']}: {w['value']}" for w in d.get("why", [])]
        return d


class TraceBundle(BaseModel):
    """Sibling bundle keyed by ``trace_id`` (same id as Audit Spine)."""

    model_config = ConfigDict(frozen=True)

    trace_id: str
    signal_snapshot: SignalSnapshotV1 | None = None
    gate_decision: GateDecisionV1 | None = None
    decision_trace: DecisionTraceV2 | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "signal_snapshot_v1": self.signal_snapshot.to_jsonable()
            if self.signal_snapshot
            else None,
            "gate_decision_v1": self.gate_decision.to_jsonable() if self.gate_decision else None,
            "decision_trace_v2": self.decision_trace.to_jsonable() if self.decision_trace else None,
        }
