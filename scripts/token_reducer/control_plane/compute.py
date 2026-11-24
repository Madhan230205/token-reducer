"""Signal fusion, gating, and decision-trace emission from AuditSpine + ContextRunState."""

from __future__ import annotations

from ..audit_spine import AuditSpine
from ..context_intelligence.models import ContextDecision
from ..intent import detect_intent
from ..orchestrator import ContextRunState
from .models import (
    ChunkRefV1,
    CompressionAggCap,
    CompressionTrace,
    DecisionTraceRetrieval,
    DecisionTraceV2,
    ExplainabilityTier,
    GateDecisionV1,
    IntentTask,
    ReliabilityTrace,
    RetrievalStats,
    SignalSnapshotV1,
    TraceBundle,
)
from .why_lines import build_why_lines, validate_why_lines


def _weak_retrieval_spine(spine: AuditSpine) -> bool:
    """Mirror orchestrator weak-pool heuristic using spine retrieval candidates."""
    cands = spine.retrieval.candidates
    if not cands:
        return True
    top = float(cands[0].score)
    if len(cands) < 3 and top < 0.22:
        return True
    return top < 0.13


def map_intent_task(query: str, routing_tier: str) -> IntentTask:
    """Finite vocabulary from StructuredIntent.type + execution tier (spec mapping table)."""
    if routing_tier == "tool":
        return "tool_execution"
    if routing_tier == "complex":
        return "multi_hop_reasoning"
    si = detect_intent(query or "")
    st = si["type"]
    if st == "code":
        return "semantic_code_search"
    if st == "analysis":
        return "qa_lookup"
    if st == "chat":
        return "general"
    return "general"


def compute_signal_snapshot(
    trace_id: str,
    *,
    audit_spine: AuditSpine | None = None,
    context_decision: ContextDecision | None = None,
) -> SignalSnapshotV1:
    """Fuse spine + context intelligence into a normalized snapshot."""
    if audit_spine is None:
        return SignalSnapshotV1(trace_id=trace_id)

    disc = audit_spine.retrieval.disagreement
    overlap = float(disc.fts_vector_overlap)
    disagreement_scalar = round(max(0.0, min(1.0, 1.0 - overlap)), 4)
    weak = _weak_retrieval_spine(audit_spine)
    cands = audit_spine.retrieval.candidates
    top_score = float(cands[0].score) if cands else 0.0
    coverage = round(min(1.0, len(cands) / 50.0), 4)
    conf = float(audit_spine.outcome.final_confidence)
    if conf <= 0.0 and context_decision is not None:
        conf = float(context_decision.confidence)
    risk = "medium"
    if context_decision is not None:
        risk = str(context_decision.risk_level)
    return SignalSnapshotV1(
        trace_id=audit_spine.trace_id,
        confidence=round(conf, 4),
        risk_level=risk,  # type: ignore[arg-type]
        disagreement=disagreement_scalar,
        weak_retrieval=weak,
        retrieval_stats=RetrievalStats(top_k_score=round(top_score, 4), coverage=coverage),
    )


def compute_gate_decision(
    snapshot: SignalSnapshotV1,
    *,
    policy_profile: str = "default",
) -> GateDecisionV1:
    """Derive allowed actions from ``snapshot`` (deterministic v1 policy)."""
    _ = policy_profile
    allow_vector = not (snapshot.risk_level == "high" and snapshot.weak_retrieval)
    allow_retry = snapshot.weak_retrieval or snapshot.disagreement >= 0.35
    cap: CompressionAggCap
    if snapshot.risk_level == "high":
        cap = "low"
    elif snapshot.risk_level == "medium" or snapshot.confidence < 0.55:
        cap = "medium"
    else:
        cap = "high"
    tier_exp: ExplainabilityTier
    if snapshot.confidence < 0.42:
        tier_exp = "minimal"
    elif snapshot.weak_retrieval:
        tier_exp = "standard"
    else:
        tier_exp = "full"
    return GateDecisionV1(
        trace_id=snapshot.trace_id,
        allow_vector=allow_vector,
        allow_retry=allow_retry,
        compression_aggressiveness_cap=cap,
        cache_reuse_allowed=True,
        explainability_tier=tier_exp,
    )


def derive_reliability(snapshot: SignalSnapshotV1, spine: AuditSpine) -> ReliabilityTrace:
    """Single implementation for trust flags (reuse from explainability if needed)."""
    disc = spine.retrieval.disagreement
    high_d = bool(disc.mismatch_flag or disc.low_overlap_flag)
    low_c = snapshot.confidence < 0.45
    return ReliabilityTrace(
        low_confidence=low_c,
        weak_retrieval=snapshot.weak_retrieval,
        high_disagreement=high_d,
    )


def _compression_preset_from_cap(cap: CompressionAggCap) -> str:
    return {"low": "conservative", "medium": "balanced", "high": "aggressive"}.get(cap, "balanced")


def _strategy_id(run_state: ContextRunState | None) -> str | None:
    if run_state is None or not run_state.context_strategy:
        return None
    cs = run_state.context_strategy
    if isinstance(cs, dict):
        sid = cs.get("strategy_id")
        return str(sid) if sid else None
    return None


def _chunk_id_to_path(spine: AuditSpine) -> dict[str, str]:
    return {str(c.chunk_id): str(c.path) for c in spine.retrieval.candidates}


def emit_decision_trace(
    trace_id: str,
    *,
    snapshot: SignalSnapshotV1 | None = None,
    gate: GateDecisionV1 | None = None,
    audit_spine: AuditSpine | None = None,
    run_state: ContextRunState | None = None,
    cache_used: bool = False,
    query: str = "",
) -> DecisionTraceV2:
    """Emit reviewer-facing trace (projection — not a second source of truth)."""
    if audit_spine is None or snapshot is None or gate is None:
        return DecisionTraceV2(
            trace_id=trace_id,
            intent_task="unknown",
            path="UNKNOWN → PACKAGE",
            compression=CompressionTrace(preset="balanced", retention_estimate=0.0, flags=()),
            confidence=0.0,
            cache_used=cache_used,
            headline_risk_level="medium",
            headline_explainability_tier="full",
        )

    spine = audit_spine
    disc = spine.retrieval.disagreement
    reliability = derive_reliability(snapshot, spine)

    parts: list[str] = []
    parts.append("CACHE_HIT" if cache_used else "CACHE_MISS")
    parts.append("RETRIEVE")
    retry_done = bool(run_state and run_state.retrieval_retry_done)
    if retry_done:
        parts.append("RETRY")
    parts.extend(["SELECT", "COMPRESS", "PACKAGE"])
    path = " → ".join(parts)

    id_to_path = _chunk_id_to_path(spine)

    omitted_ids: set[str] = set()
    if run_state is not None:
        omitted_ids = {str(o.chunk_id) for o in run_state.omitted_redundant}

    selected_list: list[ChunkRefV1] = []
    for i, cid in enumerate(spine.compression.retained_chunks):
        cid_s = str(cid)
        reason = "high relevance" if i < 3 else "selected"
        selected_list.append(
            ChunkRefV1(chunk_id=cid_s, path=id_to_path.get(cid_s, ""), reason=reason)
        )

    drop_list: list[ChunkRefV1] = []
    for cid in spine.compression.dropped_chunks:
        cid_s = str(cid)
        reason = "redundant" if cid_s in omitted_ids else "pruned"
        drop_list.append(ChunkRefV1(chunk_id=cid_s, path=id_to_path.get(cid_s, ""), reason=reason))

    flags: list[str] = []
    if disc.mismatch_flag or disc.low_overlap_flag:
        flags.append("disagreement_present")
    if snapshot.weak_retrieval:
        flags.append("weak_retrieval")

    preset = _compression_preset_from_cap(gate.compression_aggressiveness_cap)
    retention = round(float(spine.compression.loss_metrics.importance_retained_ratio), 4)
    conf = round(float(spine.outcome.final_confidence), 4)

    cands = spine.retrieval.candidates
    top_score = round(float(cands[0].score), 4) if cands else 0.0
    retr = DecisionTraceRetrieval(
        top_score=top_score,
        n_candidates=len(cands),
        fts_vector_overlap=round(float(disc.fts_vector_overlap), 4),
    )

    routing_tier = spine.routing.selected_tier
    tier = routing_tier
    tool_skill: str | None = None
    subagents_used: tuple[str, ...] = ()
    compression_triggered = bool(spine.compression.output_tokens > 0)
    if run_state is not None:
        compression_triggered = bool(run_state.bullets)
        er = run_state.execution_route
        if er is not None and er.tier == "tool":
            tool_skill = er.skill_id
        sub_meta = run_state.subagent_debug if isinstance(run_state.subagent_debug, dict) else {}
        ran = sub_meta.get("subagents")
        if isinstance(ran, list):
            subagents_used = tuple(str(x) for x in ran)

    q = query or (run_state.query if run_state is not None else "")
    intent_task = map_intent_task(q, routing_tier)
    strat = _strategy_id(run_state)

    why_t = build_why_lines(
        intent_task=intent_task,
        snapshot=snapshot,
        gate=gate,
        spine=spine,
        compression_preset=preset,
        retention_estimate=retention,
        compression_flags=tuple(flags),
        retrieval_retry_done=retry_done,
    )
    validate_why_lines(why_t)

    return DecisionTraceV2(
        trace_id=spine.trace_id,
        intent_task=intent_task,
        strategy_id=strat,
        tier=tier,
        routing_tier=routing_tier,
        path=path,
        selected_chunks=tuple(selected_list),
        dropped_chunks=tuple(drop_list),
        why=why_t,
        compression=CompressionTrace(
            preset=preset,
            retention_estimate=retention,
            flags=tuple(flags),
        ),
        confidence=conf,
        cache_used=cache_used,
        reliability=reliability,
        retrieval=retr,
        headline_disagreement=round(float(snapshot.disagreement), 4),
        headline_weak_retrieval=snapshot.weak_retrieval,
        headline_risk_level=snapshot.risk_level,  # type: ignore[arg-type]
        headline_explainability_tier=gate.explainability_tier,
        headline_gate_allow_vector=gate.allow_vector,
        headline_gate_allow_retry=gate.allow_retry,
        headline_gate_compression_cap=gate.compression_aggressiveness_cap,
        retrieval_retry_triggered=retry_done,
        tool_skill_id=tool_skill,
        subagents_used=subagents_used,
        compression_triggered=compression_triggered,
    )


def build_trace_bundle_skeleton(
    trace_id: str,
    *,
    audit_spine: AuditSpine | None = None,
    context_decision: ContextDecision | None = None,
    run_state: ContextRunState | None = None,
    cache_used: bool = False,
    query: str = "",
) -> TraceBundle:
    """Build TraceBundle; uses real projections when ``audit_spine`` is set."""
    if audit_spine is not None:
        tid = audit_spine.trace_id
        cd = context_decision
        if run_state is not None and cd is None:
            cd = run_state.context_decision
        q = query or (run_state.query if run_state is not None else "")
        snap = compute_signal_snapshot(tid, audit_spine=audit_spine, context_decision=cd)
        gate = compute_gate_decision(snap)
        dtrace = emit_decision_trace(
            tid,
            snapshot=snap,
            gate=gate,
            audit_spine=audit_spine,
            run_state=run_state,
            cache_used=cache_used,
            query=q,
        )
        return TraceBundle(trace_id=tid, signal_snapshot=snap, gate_decision=gate, decision_trace=dtrace)

    snap = compute_signal_snapshot(trace_id)
    gate = compute_gate_decision(snap)
    dtrace = emit_decision_trace(
        trace_id,
        snapshot=snap,
        gate=gate,
        audit_spine=None,
        cache_used=False,
        query=query,
    )
    return TraceBundle(trace_id=trace_id, signal_snapshot=snap, gate_decision=gate, decision_trace=dtrace)


def build_trace_bundle_from_run(
    state: ContextRunState,
    *,
    cache_used: bool = False,
) -> TraceBundle | None:
    """Build a full TraceBundle from a completed pipeline run."""
    spine = state.audit_spine
    if spine is None:
        return None
    return build_trace_bundle_skeleton(
        spine.trace_id,
        audit_spine=spine,
        context_decision=state.context_decision,
        run_state=state,
        cache_used=cache_used,
        query=state.query,
    )
