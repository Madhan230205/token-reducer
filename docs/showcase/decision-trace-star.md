# Decision trace — sample JSON, UX, and spine mapping

**Goal:** Make `decision_trace_v2` (and its TraceBundle siblings) the **default mental model** for “why this context.”

**Specs:** [TraceBundle / control plane](../superpowers/specs/2026-05-02-trace-bundle-control-plane-design.md) · [Audit spine](../superpowers/specs/2026-05-02-audit-spine-design.md) · [Hybrid explainability UX](../superpowers/specs/2026-05-02-plugin-explainability-hybrid-ux-design.md)

> All JSON below is **illustrative** (consistent with schemas). Replace `trace_id` with a live UUID from a run.

---

## 1. Sample `decision_trace_v2` (public gold)

This is the **reviewer- and product-facing** summary: `selected_chunks` / `dropped_chunks` (id + path + reason), structured **`why`**, **`intent_task`**, headline snapshot/gate fields, compression, path, cache.

```json
{
  "schema_version": "decision_trace_v2",
  "trace_id": "8f2c1a9e-4b0d-4a7e-9c12-0a1b2c3d4e5f",
  "intent_task": "semantic_code_search",
  "strategy_id": "behavior_truth",
  "tier": "simple",
  "routing_tier": "simple",
  "path": "CACHE_MISS → RETRIEVE → MERGE → RERANK → SELECT → SUBAGENTS → COMPRESS → PACKAGE",
  "selected_chunks": [
    { "chunk_id": "c_1042", "path": "src/auth.py", "reason": "top relevance" },
    { "chunk_id": "c_1048", "path": "src/session.ts", "reason": "patch path" }
  ],
  "dropped_chunks": [
    { "chunk_id": "c_1101", "path": "tests/test_auth.py", "reason": "low score" },
    { "chunk_id": "c_1102", "path": "legacy/old.py", "reason": "redundant" }
  ],
  "why": [
    { "category": "intent", "value": "task semantic_code_search" },
    { "category": "routing", "value": "retrieval tier simple (risk_flags=none)" }
  ],
  "why_display": [
    "intent: task semantic_code_search",
    "routing: retrieval tier simple (risk_flags=none)"
  ],
  "compression": {
    "preset": "balanced",
    "retention_estimate": 0.64,
    "flags": ["disagreement_present"]
  },
  "confidence": 0.78,
  "cache_used": false,
  "headline_disagreement": 0.18,
  "headline_weak_retrieval": false,
  "headline_risk_level": "medium",
  "headline_explainability_tier": "full",
  "headline_gate_allow_vector": true,
  "headline_gate_allow_retry": true,
  "headline_gate_compression_cap": "medium",
  "retrieval_retry_triggered": false
}
```

**Siblings (same `trace_id`, not embedded in spine):**

```json
{
  "trace_id": "8f2c1a9e-4b0d-4a7e-9c12-0a1b2c3d4e5f",
  "signal_snapshot_v1": {
    "schema_version": "signal_snapshot_v1",
    "trace_id": "8f2c1a9e-4b0d-4a7e-9c12-0a1b2c3d4e5f",
    "confidence": 0.72,
    "risk_level": "medium",
    "disagreement": 0.31,
    "weak_retrieval": true,
    "retrieval_stats": { "top_k_score": 0.61, "coverage": 0.48 }
  },
  "gate_decision_v1": {
    "schema_version": "gate_decision_v1",
    "trace_id": "8f2c1a9e-4b0d-4a7e-9c12-0a1b2c3d4e5f",
    "allow_vector": true,
    "allow_retry": true,
    "compression_aggressiveness_cap": "medium",
    "cache_reuse_allowed": false,
    "explainability_tier": "full"
  }
}
```

`TraceBundle.to_jsonable()` keys the decision trace as **`decision_trace_v2`**.

---

## 2. Rendered UX (hybrid — what users see)

Hybrid one-liner and sidecar are **views** from the same spine + run inputs as `emit_decision_trace` — see [hybrid UX spec](../superpowers/specs/2026-05-02-plugin-explainability-hybrid-ux-design.md).

**Model-facing (one line, ~15 tokens)** — field: `context_optimization_line`

```text
Context optimized (balanced, medium confidence). Key focus: auth.py + session.ts.
```

**User-facing sidecar (3–6 lines)** — field: `context_explainability.sidecar`

```text
Selected: auth.py (top relevance), session.ts (patch path)
Dropped: test_auth.py (redundant)
Compression: balanced • medium confidence
```

**Canonical JSON for reviewers** — `claude_context.decision_trace` / `ContextPacket.decision_trace` / `decision_trace.json` on disk.

**Verbose** (`TOKEN_REDUCER_EXPLAIN_VERBOSE=1`): full `audit_spine` JSON + counts — not shown by default.

---

## 3. How it maps to the Audit Spine

The spine is the **canonical factual record** per stage; the decision trace is a **curated projection** for humans and reviewers.

| Decision trace field | Spine section(s) | Notes |
|---------------------|------------------|--------|
| `path` | Derived from pipeline + cache | Encode major transitions (cache hit/miss, retry, compress). |
| `selected_chunks[]` | `compression.retained_chunks` + `retrieval.candidates` | Join chunk id → path via candidate rows; reasons from selection logic. |
| `dropped_chunks[]` | `compression.dropped_chunks` | Reasons bucketed (redundant, pruned). |
| `why[]` | Snapshot + gate + spine slices | Closed categories; see `control_plane/why_lines.py`. |
| `intent_task` | `StructuredIntent` + `routing.selected_tier` | Finite enum mapping in `compute.map_intent_task`. |
| `compression.preset` | Gate cap + policy | Maps to conservative / balanced / aggressive UX language. |
| `compression.retention_estimate` | `compression.loss_metrics.importance_retained_ratio` | Rounded. |
| `compression.flags` | `retrieval.disagreement`, snapshot | e.g. `disagreement_present`. |
| `confidence` | `outcome.final_confidence` | Headline number. |
| Headline duplicates | Snapshot + gate | So `decision_trace.json` stands alone without opening siblings. |

**Rule:** Do not invent trace rows that cannot be justified by spine + gate + snapshot. The trace is a **view**, not a second brain.

---

## 4. Where this lives in code (today)

- **Types:** `scripts/token_reducer/control_plane/models.py` (`DecisionTraceV2`, `TraceBundle`, `WhyLineStruct`, …).
- **Emitters:** `scripts/token_reducer/control_plane/compute.py` (`emit_decision_trace`, `build_trace_bundle_from_run`, `map_intent_task`).
- **Why lines:** `scripts/token_reducer/control_plane/why_lines.py`.
- **Spine:** `scripts/token_reducer/audit_spine.py`.
- **Hybrid UX renderer:** `scripts/token_reducer/explainability_render.py`.
- **Export / summary:** `scripts/token_reducer/trace_export.py`.

Plugin and packet attach **`decision_trace`** after the run (`pipeline.py`); **`summary.txt`** follows the same structured fields.
