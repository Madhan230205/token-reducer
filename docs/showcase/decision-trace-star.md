# Decision trace — sample JSON, UX, and spine mapping

**Goal:** Make `decision_trace_v1` (and its TraceBundle siblings) the **default mental model** for “why this context.”

**Specs:** [TraceBundle / control plane](../superpowers/specs/2026-05-02-trace-bundle-control-plane-design.md) · [Audit spine](../superpowers/specs/2026-05-02-audit-spine-design.md) · [Hybrid explainability UX](../superpowers/specs/2026-05-02-plugin-explainability-hybrid-ux-design.md)

> All JSON below is **illustrative** (consistent with schemas). Replace `trace_id` with a live UUID from a run.

---

## 1. Sample `decision_trace_v1` (public gold)

This is the **reviewer- and product-facing** summary: kept/dropped, compression, path, cache.

```json
{
  "schema_version": "decision_trace_v1",
  "trace_id": "8f2c1a9e-4b0d-4a7e-9c12-0a1b2c3d4e5f",
  "path": "CACHE_MISS → RETRIEVE → MERGE → RERANK → SELECT → SUBAGENTS → COMPRESS → PACKAGE",
  "kept": [
    { "id": "c_1042", "reason": "top relevance" },
    { "id": "c_1048", "reason": "patch path" }
  ],
  "dropped": [
    { "id": "c_1101", "reason": "low score" },
    { "id": "c_1102", "reason": "redundant" }
  ],
  "compression": {
    "preset": "balanced",
    "retention_estimate": 0.64,
    "flags": ["disagreement_present"]
  },
  "confidence": 0.78,
  "cache_used": false
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

---

## 2. Rendered UX (hybrid — what users see)

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

**Verbose** (`TOKEN_REDUCER_EXPLAIN_VERBOSE=1`): full `audit_spine` JSON + counts — not shown by default.

---

## 3. How it maps to the Audit Spine

The spine is the **canonical factual record** per stage; the decision trace is a **curated projection** for humans and reviewers.

| Decision trace field | Spine section(s) | Notes |
|---------------------|------------------|--------|
| `path` | Derived from pipeline + cache | Encode major transitions (cache hit/miss, retry, compress). |
| `kept[].id` | `compression.retained_chunks` + `retrieval.candidates` | Join chunk id → path/reason via candidate rows + selection logic. |
| `dropped[].id` | `compression.dropped_chunks` (+ pool-not-selected later) | Reasons bucketed (redundant, low score, prune). |
| `compression.preset` | Routing / strategy / profile | Maps to “balanced / fast” UX preset. |
| `compression.retention_estimate` | `compression.loss_metrics.importance_retained_ratio` (or successor) | Same number family; trace uses rounded language. |
| `compression.flags` | `retrieval.disagreement`, `routing.risk_flags` | e.g. `disagreement_present` ↔ `mismatch_flag` / low overlap. |
| `confidence` | `outcome.final_confidence` (+ intel adjustments) | Single headline number for trace. |
| `cache_used` | Packet / pipeline cache branch | Not currently a spine root field; bridge from `ContextPacket.cache.hit` when wiring. |

**Rule:** Do not invent trace rows that cannot be justified by spine + gate + snapshot. The trace is a **view**, not a second brain.

---

## 4. Where this lives in code (today)

- **Types:** `scripts/token_reducer/control_plane/models.py` (`DecisionTraceV1`, `TraceBundle`, …).
- **Stubs (wire next):** `scripts/token_reducer/control_plane/compute.py` (`emit_decision_trace`, …).
- **Spine:** `scripts/token_reducer/audit_spine.py`.
- **Hybrid UX renderer:** `scripts/token_reducer/explainability_render.py`.

Next implementation step: populate `emit_decision_trace` from real `AuditSpine` + selection metadata, then attach `TraceBundle.to_jsonable()` to the plugin payload or harness artifact alongside `audit_spine`.
