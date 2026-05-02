"""Canonical Audit Spine (v0.1) — single decision record per `docs/superpowers/specs/2026-05-02-audit-spine-design.md`."""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .context_intelligence.models import ContextDecision
from .execution_route import ExecutionRoute
from .intent import StructuredIntent
from .models import Candidate, OmittedRedundantEntry, hash_text

SCHEMA_VERSION = "audit_spine_v0_1"

ExecutionTier = Literal["simple", "tool", "complex"]
RetrievalSource = Literal["fts", "vector", "hybrid"]


class QueryFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    length: int = 0
    type: ExecutionTier = "simple"


class QuerySection(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw: str = ""
    normalized: str = ""
    features: QueryFeatures = Field(default_factory=QueryFeatures)


class RetrievalCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    chunk_id: str
    source: RetrievalSource
    score: float = 0.0
    rank: int = 0


class RetrievalDisagreement(BaseModel):
    model_config = ConfigDict(frozen=True)

    fts_vector_overlap: float = 1.0
    rank_correlation: float = 1.0
    low_overlap_flag: bool = False
    mismatch_flag: bool = False


class RetrievalSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidates: tuple[RetrievalCandidate, ...] = ()
    disagreement: RetrievalDisagreement = Field(default_factory=RetrievalDisagreement)


class RoutingThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    confidence: float = 0.0
    risk: float = 0.0


class RoutingSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    selected_tier: ExecutionTier = "simple"
    thresholds: RoutingThresholds = Field(default_factory=RoutingThresholds)
    risk_flags: tuple[str, ...] = ()
    decision_reason: str = ""


class CompressionLossMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    importance_retained_ratio: float = 0.0
    salient_span_checksum: str = ""


class CompressionSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    retained_chunks: tuple[str, ...] = ()
    dropped_chunks: tuple[str, ...] = ()
    loss_metrics: CompressionLossMetrics = Field(default_factory=CompressionLossMetrics)


class OutcomeSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    latency_ms: int = 0
    final_confidence: float = 0.0


class AdaptiveSignals(BaseModel):
    model_config = ConfigDict(frozen=True)

    retrieval_usefulness: float | None = None
    compression_loss: float | None = None


class AdaptiveUpdate(BaseModel):
    model_config = ConfigDict(frozen=True)

    parameter: str = ""
    delta: float = 0.0
    reason: str = ""


class AdaptiveSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    signals: AdaptiveSignals = Field(default_factory=AdaptiveSignals)
    updates: tuple[AdaptiveUpdate, ...] = ()


class AuditSpine(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    trace_id: str
    timestamp: str
    query: QuerySection = Field(default_factory=QuerySection)
    retrieval: RetrievalSection = Field(default_factory=RetrievalSection)
    routing: RoutingSection = Field(default_factory=RoutingSection)
    compression: CompressionSection = Field(default_factory=CompressionSection)
    outcome: OutcomeSection = Field(default_factory=OutcomeSection)
    adaptive: AdaptiveSection = Field(default_factory=AdaptiveSection)

    def to_jsonable(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def new_audit_spine() -> tuple[AuditSpine, str, str]:
    """Fresh spine with ``trace_id`` and ISO timestamp."""
    trace_id = str(uuid.uuid4())
    ts = datetime.now(UTC).isoformat()
    return (
        AuditSpine(
            trace_id=trace_id,
            timestamp=ts,
        ),
        trace_id,
        ts,
    )


def bootstrap_spine(query: str, si: StructuredIntent, route: ExecutionRoute | None) -> AuditSpine:
    """Root metadata + query slice before retrieval."""
    spine, _, _ = new_audit_spine()
    return spine.model_copy(update={"query": build_query_section(query, si, route)})


def build_query_section(
    query: str,
    si: StructuredIntent,
    route: ExecutionRoute | None,
) -> QuerySection:
    """Initial query slice (routing tier seeds ``features.type`` for v0.1)."""
    tier: ExecutionTier = route.tier if route is not None else "simple"
    _ = si
    return QuerySection(
        raw=query,
        normalized=query.strip()[:4000],
        features=QueryFeatures(
            length=len(query.split()),
            type=tier,
        ),
    )


def _candidate_retrieval_source(c: Candidate) -> RetrievalSource:
    has_v = (c.vector_rank is not None) or (c.vector_score > 1e-9)
    has_f = (c.fts_rank is not None) or (c.bm25_score is not None) or (c.fts_score > 1e-9)
    if has_v and has_f:
        return "hybrid"
    if has_v:
        return "vector"
    return "fts"


def scored_pool_to_candidates(scored: list[Candidate], *, top_n: int = 50) -> tuple[RetrievalCandidate, ...]:
    out: list[RetrievalCandidate] = []
    for rank, c in enumerate(scored[:top_n], start=1):
        src = _candidate_retrieval_source(c)
        out.append(
            RetrievalCandidate(
                doc_id=hash_text(str(c.source))[:24],
                chunk_id=str(c.chunk_id),
                source=src,
                score=float(c.final_score),
                rank=rank,
            )
        )
    return tuple(out)


def _rank_avg(values: list[float]) -> list[float]:
    """Average ranks for ``values`` order (1-based), ties resolved."""
    n = len(values)
    if n == 0:
        return []
    enum = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        v0 = enum[i][1]
        while j + 1 < n and enum[j + 1][1] == v0:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            orig = enum[k][0]
            ranks[orig] = avg
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 1.0
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in a)
    db = sum((y - mb) ** 2 for y in b)
    if da <= 1e-12 or db <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, num / math.sqrt(da * db)))


def compute_retrieval_disagreement(
    fts_hits: list[Candidate],
    vector_hits: list[Candidate],
    *,
    top_n: int = 32,
) -> RetrievalDisagreement:
    """Single place for FTS vs vector disagreement (spec §4.1)."""
    fts_ids = [c.chunk_id for c in fts_hits[:top_n]]
    vec_ids = [c.chunk_id for c in vector_hits[:top_n]]
    if not vector_hits:
        return RetrievalDisagreement(
            fts_vector_overlap=1.0,
            rank_correlation=1.0,
            low_overlap_flag=False,
            mismatch_flag=False,
        )
    set_f, set_v = set(fts_ids), set(vec_ids)
    inter = set_f & set_v
    union = set_f | set_v
    overlap = len(inter) / len(union) if union else 0.0

    fts_ranks = {c.chunk_id: i + 1 for i, c in enumerate(fts_hits[:top_n])}
    vec_ranks = {c.chunk_id: i + 1 for i, c in enumerate(vector_hits[:top_n])}
    common = sorted(inter, key=lambda x: fts_ranks.get(x, 9999))
    if len(common) < 2:
        corr = 1.0 if not common else 0.0
    else:
        fr = [float(fts_ranks[cid]) for cid in common]
        vr = [float(vec_ranks[cid]) for cid in common]
        r1 = _rank_avg(fr)
        r2 = _rank_avg(vr)
        corr = _pearson(r1, r2)

    low_overlap = overlap < 0.12 and len(union) > 0
    mismatch = low_overlap or (len(common) >= 2 and corr < 0.25)
    return RetrievalDisagreement(
        fts_vector_overlap=round(overlap, 4),
        rank_correlation=round(corr, 4),
        low_overlap_flag=low_overlap,
        mismatch_flag=mismatch,
    )


def _risk_scalar(risk_level: str) -> float:
    return {"low": 0.2, "medium": 0.5, "high": 0.8}.get(risk_level, 0.5)


def build_routing_section(
    route: ExecutionRoute | None,
    disagreement: RetrievalDisagreement,
    context_decision: ContextDecision | None,
) -> RoutingSection:
    tier: ExecutionTier = route.tier if route is not None else "simple"
    flags: list[str] = []
    if disagreement.low_overlap_flag:
        flags.append("low_overlap")
    if disagreement.mismatch_flag:
        flags.append("high_disagreement")
    conf = 0.5
    risk = _risk_scalar("medium")
    thr_conf = 0.42
    if context_decision is not None:
        conf = float(context_decision.confidence)
        thr_conf = float(context_decision.confidence_threshold_used)
        risk = _risk_scalar(str(context_decision.risk_level))
    reason_parts = [
        f"execution_tier={tier}",
        f"fts_vector_overlap={disagreement.fts_vector_overlap}",
        f"rank_correlation={disagreement.rank_correlation}",
    ]
    if context_decision is not None:
        reason_parts.append(f"intel_confidence={conf:.3f}")
        reason_parts.append(f"confidence_gate={thr_conf:.3f}")
    reason = "; ".join(reason_parts)[:900]
    return RoutingSection(
        selected_tier=tier,
        thresholds=RoutingThresholds(
            confidence=round(conf, 4),
            risk=round(risk, 4),
        ),
        risk_flags=tuple(flags),
        decision_reason=reason,
    )


def apply_retrieval_and_routing(
    spine: AuditSpine,
    *,
    scored_pool: list[Candidate],
    fts_hits: list[Candidate],
    vector_hits: list[Candidate],
    route: ExecutionRoute | None,
    context_decision: ContextDecision | None,
) -> AuditSpine:
    """Immutably replace ``retrieval`` and ``routing`` after the retriever phase."""
    disc = compute_retrieval_disagreement(fts_hits, vector_hits)
    cands = scored_pool_to_candidates(scored_pool, top_n=50)
    routing = build_routing_section(route, disc, context_decision)
    return spine.model_copy(
        update={
            "retrieval": RetrievalSection(candidates=cands, disagreement=disc),
            "routing": routing,
        }
    )


def apply_compression_slice(
    spine: AuditSpine,
    *,
    selected: list[Candidate],
    omitted: list[OmittedRedundantEntry],
    bullets: list[str],
) -> AuditSpine:
    """Record compression / selection outcomes after the compressor phase."""
    pre_tok = sum(int(c.token_estimate) for c in selected)
    out_tok = sum(int(max(1, len(b.split()) * 4 // 3)) for b in bullets)  # rough token proxy
    retained = tuple(str(c.chunk_id) for c in selected)
    dropped = [str(o.chunk_id) for o in omitted]
    ratio = min(1.0, out_tok / pre_tok) if pre_tok > 0 else 1.0
    checksum = hash_text("\n".join(bullets))[:48] if bullets else ""
    comp = CompressionSection(
        input_tokens=pre_tok,
        output_tokens=out_tok,
        retained_chunks=retained,
        dropped_chunks=tuple(dropped),
        loss_metrics=CompressionLossMetrics(
            importance_retained_ratio=round(ratio, 4),
            salient_span_checksum=checksum,
        ),
    )
    return spine.model_copy(update={"compression": comp})


def apply_outcome(
    spine: AuditSpine,
    *,
    latency_ms: int,
    context_decision: ContextDecision | None,
    disagreement: RetrievalDisagreement | None,
) -> AuditSpine:
    base_conf = float(context_decision.confidence) if context_decision is not None else 0.75
    penalty = 0.0
    if disagreement is not None:
        if disagreement.mismatch_flag:
            penalty += 0.12
        if disagreement.low_overlap_flag:
            penalty += 0.06
    final_c = max(0.0, min(1.0, base_conf - penalty))
    out = OutcomeSection(latency_ms=max(0, int(latency_ms)), final_confidence=round(final_c, 4))
    return spine.model_copy(update={"outcome": out})
