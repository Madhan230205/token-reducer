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
| **DecisionTrace** | **Public-friendly gold path** — structured outcome for reviewers and UX: chunk refs, compression, cache path, categorized **why** lines, **intent_task** (finite vocabulary). |

Merging GateDecision into the spine **blurs policy vs outcome**, weakens **replay under alternate policies**, and complicates **debugging**.

All siblings share **`trace_id`** with the Audit Spine root.

**Canonical JSON:** `decision_trace_v2` is the **single** reviewer-facing artifact on disk and in plugin/packet output. Hybrid explainability (one-liner + sidecar) and `summary.txt` are **deterministic views** built from the **same** spine + snapshot + gate inputs as `emit_decision_trace` — not parallel sources of truth.

---

## 2. TraceBundle layout

```text
TraceBundle
├── signal_snapshot_v1    # truth-ish inputs to policy
├── gate_decision_v1      # allowed actions / caps
├── decision_trace_v2     # reviewer-facing outcome summary (canonical)
```

Optional future: `adapt_feedback_events_v1` (append-only) — out of scope here.

---

## 3. Schema sketches (v1 siblings + v2 decision trace)

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

### 3.3 `decision_trace_v2`

**Breaking change from v1:** `kept` / `dropped` are removed. Use **`selected_chunks`** and **`dropped_chunks`** only, each `{ chunk_id, path, reason }`. **`why`** is an array of structured objects `{ category, value }` from a **closed category set** (no free-form prose). **`intent_task`** is a finite enum — not arbitrary text.

`decision_trace.json` MUST be **readable alone** for reviewers: headline duplicates of snapshot/gate signals (`headline_disagreement`, `headline_weak_retrieval`, `headline_risk_level`, gate allow flags and cap, `retrieval_retry_triggered`, etc.) sit on the trace object in addition to nested `reliability` and `retrieval`.

```json
{
  "schema_version": "decision_trace_v2",
  "trace_id": "...",
  "intent_task": "semantic_code_search",
  "strategy_id": "behavior_truth",
  "tier": "simple",
  "routing_tier": "simple",
  "path": "CACHE_MISS → RETRIEVE → SELECT → COMPRESS → PACKAGE",
  "selected_chunks": [
    {"chunk_id": "42", "path": "src/app.py", "reason": "high relevance"}
  ],
  "dropped_chunks": [
    {"chunk_id": "7", "path": "legacy/x.py", "reason": "pruned"}
  ],
  "why": [
    {"category": "routing", "value": "retrieval tier simple (risk_flags=none)"},
    {"category": "weak_retrieval", "value": "false"}
  ],
  "why_display": ["routing: retrieval tier simple (risk_flags=none)", "weak_retrieval: false"],
  "compression": {
    "preset": "balanced",
    "retention_estimate": 0.64,
    "flags": ["disagreement_present"]
  },
  "confidence": 0.78,
  "cache_used": false,
  "reliability": {
    "low_confidence": false,
    "weak_retrieval": false,
    "high_disagreement": false
  },
  "retrieval": {
    "top_score": 0.61,
    "n_candidates": 12,
    "fts_vector_overlap": 0.82
  },
  "headline_disagreement": 0.18,
  "headline_weak_retrieval": false,
  "headline_risk_level": "medium",
  "headline_explainability_tier": "full",
  "headline_gate_allow_vector": true,
  "headline_gate_allow_retry": true,
  "headline_gate_compression_cap": "medium",
  "retrieval_retry_triggered": false,
  "tool_skill_id": null,
  "subagents_used": [],
  "compression_triggered": true
}
```

**`intent_task` mapping (pipeline):** `tool_execution` if routing tier is `tool`; `multi_hop_reasoning` if tier is `complex`; else from `StructuredIntent.type`: `code` → `semantic_code_search`, `analysis` → `qa_lookup`, `chat` → `general`; missing spine → `unknown`.

**`why` categories (closed set):** `intent`, `routing`, `weak_retrieval`, `disagreement`, `retrieval`, `compression`, `retry`, `gate`, `reliability` — each line is a deterministic projection from spine / snapshot / gate (see `control_plane/why_lines.py`).

**Implemented:** `DecisionTraceV2`, `ChunkRefV1`, `WhyLineStruct` in `scripts/token_reducer/control_plane/models.py`; `emit_decision_trace` in `compute.py`.

---

## 4. Replayability contract

- **Same `signal_snapshot_v1` + different gate rules** → simulate alternate policy without re-running retrieval (offline evaluator).
- **Same `gate_decision_v1` + different execution** → detect drift or bugs in stages.
- **Spine remains** the detailed factual record; DecisionTrace remains the **curated** projection for humans and external review.

---

## 5. On-disk trace bundle layout (v1)

After a pipeline run (when `--trace-dir` is set or `TOKEN_REDUCER_TRACE_DIR` is non-empty), artifacts are written to:

`<trace_root>/<trace_id>/`

| File | Role |
|------|------|
| `manifest.json` | `schema_version`: `trace_manifest_v1`, `inputs_digest` (query hash, db path, index fingerprint, session), `files` map of `sha256:` digests |
| `audit_spine.json` | Canonical spine (`to_jsonable()`) |
| `signal_snapshot.json` | `signal_snapshot_v1` |
| `gate_decision.json` | `gate_decision_v1` |
| `decision_trace.json` | `decision_trace_v2` (public gold) |
| `summary.txt` | Deterministic human summary (template-filled; no LLM) |

**JSON serialization:** UTF-8, object keys sorted, 2-space indent, trailing newline — stable content hashing for manifest entries.

---

## 6. Phase-A replay (CLI)

Command: `token-reducer trace replay <trace_dir> [--gate-patch '<json>']`

- Validates required files and Pydantic-loads `decision_trace.json` as **`decision_trace_v2`**, `gate_decision.json`, `signal_snapshot.json` when present.
- Optional `--gate-patch` merges allowed keys onto `gate_decision_v1` and prints a field-wise diff (`before` / `after`). Full pipeline re-execution is **not** part of phase A.

---

## 7. Implementation status

- `scripts/token_reducer/control_plane/compute.py` — `compute_signal_snapshot`, `compute_gate_decision`, `emit_decision_trace`, `build_trace_bundle_from_run` derive from `AuditSpine` + `ContextRunState`.
- `scripts/token_reducer/trace_export.py` — `write_trace_bundle_dir`, `render_trace_summary`.
- `scripts/token_reducer/trace_replay.py` — `validate_trace_dir`, `summarize_replay`, gate patch merge.
- `scripts/token_reducer/pipeline.py` — optional export after hybrid explainability attach.

---

## 8. Non-goals (v0)

- Moving ANN, compression heuristics, or agent topology.
- Embedding `GateDecision` inside `AuditSpine`.
- LLM-generated traces.

---

## Document history

| Rev | Notes |
|-----|--------|
| 0.1 | Sibling artifacts, TraceBundle, replay rationale, skeleton phase. |
| 0.3 | **`decision_trace_v2`**: `selected_chunks`/`dropped_chunks`, structured `why`, `intent_task`, headline fields; hybrid UX aligns to same builder inputs. |
