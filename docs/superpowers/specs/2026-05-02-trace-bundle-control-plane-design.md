# TraceBundle — Control Plane Artifacts (Design Spec)

**Status:** Approved v0.1 — sibling artifacts linked by `trace_id`  
**Date:** 2026-05-02  
**Scope:** token-reducer — separate **signals**, **policy gates**, and **execution outcomes** from the Audit Spine so replay, debugging, and explainability stay honest.

**Related**

- `docs/superpowers/specs/2026-05-02-audit-spine-design.md` — **Audit Spine = what happened** in the pipeline record (facts); **not** the policy layer.
- `docs/superpowers/specs/2026-05-02-plugin-explainability-hybrid-ux-design.md` — human/model surfaces **project** from these artifacts + spine.
- `docs/superpowers/specs/2026-05-02-benchmark-proof-harness-design.md` — harness may consume TraceBundle for regression and replay scenarios.

---

## 1. Principle: do not embed GateDecision in the spine

| Artifact | Role |
|----------|------|
| **Audit Spine** | Canonical **execution / observation** record for the run (append-only stage slices). |
| **SignalSnapshot** | **Inputs** to policy — normalized scalar/boolean view of confidence, risk, disagreement, weak retrieval, etc. |
| **GateDecision** | **Policy interpretation** — what was **allowed** (vector, retry, compression cap, cache reuse, explainability tier). |
| **DecisionTrace** | **Public-friendly gold path** — narrative structure for kept/dropped/compression/cache path for reviewers and UX. |

Merging GateDecision into the spine **blurs policy vs outcome**, weakens **replay under alternate policies**, and complicates **debugging**.

All siblings share **`trace_id`** with the Audit Spine root.

---

## 2. TraceBundle layout

```text
TraceBundle
├── signal_snapshot_v1    # truth-ish inputs to policy
├── gate_decision_v1      # allowed actions / caps
├── decision_trace_v1     # reviewer-facing outcome summary
```

Optional future: `adapt_feedback_events_v1` (append-only) — out of scope here.

---

## 3. Schema sketches (v1)

### 3.1 `signal_snapshot_v1`

```json
{
  "schema_version": "signal_snapshot_v1",
  "trace_id": "...",
  "confidence": 0.72,
  "risk_level": "medium",
  "disagreement": 0.31,
  "weak_retrieval": true,
  "retrieval_stats": {
    "top_k_score": 0.61,
    "coverage": 0.48
  }
}
```

### 3.2 `gate_decision_v1`

```json
{
  "schema_version": "gate_decision_v1",
  "trace_id": "...",
  "allow_vector": true,
  "allow_retry": true,
  "compression_aggressiveness_cap": "medium",
  "cache_reuse_allowed": false,
  "explainability_tier": "full"
}
```

### 3.3 `decision_trace_v1`

```json
{
  "schema_version": "decision_trace_v1",
  "trace_id": "...",
  "path": "CACHE_MISS → RETRIEVE → RETRY → SELECT → COMPRESS → PACKAGE",
  "kept": [
    {"id": "c1", "reason": "high relevance"},
    {"id": "c2", "reason": "supports answer"}
  ],
  "dropped": [
    {"id": "c7", "reason": "low score"},
    {"id": "c9", "reason": "redundant"}
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

---

## 4. Replayability contract

- **Same `signal_snapshot_v1` + different gate rules** → simulate alternate policy without re-running retrieval (offline evaluator).
- **Same `gate_decision_v1` + different execution** → detect drift or bugs in stages.
- **Spine remains** the detailed factual record; DecisionTrace remains the **curated** projection for humans and external review.

---

## 5. Implementation phase (skeleton only)

Python module(s) under `scripts/token_reducer/control_plane/`:

- Typed models for the three artifacts + `TraceBundle`.
- Functions (contracts only for v0):
  - `compute_signal_snapshot(...)`
  - `compute_gate_decision(...)`
  - `emit_decision_trace(...)`
- **No change** to retrieval/compression/agents in v0 — stubs return structurally valid defaults tied to `trace_id` until wired.

---

## 6. Non-goals (v0)

- Moving ANN, compression heuristics, or agent topology.
- Embedding `GateDecision` inside `AuditSpine`.
- LLM-generated traces.

---

## Document history

| Rev | Notes |
|-----|--------|
| 0.1 | Sibling artifacts, TraceBundle, replay rationale, skeleton phase. |
