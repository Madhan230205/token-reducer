# Audit Spine — Canonical Decision Record (Design Spec)

**Status:** Draft v0.1 — schema and stage contracts for review  
**Date:** 2026-05-02  
**Scope:** token-reducer — single source of truth for **routing**, **disagreement**, **compression**, and **adaptive** reasoning; powers UX traces, CI/benchmark replay, and bounded learning.

**Related**

- `docs/superpowers/specs/2026-05-02-context-intelligence-layer-design.md` — bounds and caps that the spine must respect where overlap exists.
- `docs/superpowers/specs/2026-05-02-benchmark-proof-harness-design.md` — replay, thresholds, stability; harness consumes **full spine** (this spec).
- `docs/superpowers/specs/2026-05-02-adaptive-feedback-loop-design.md` — adaptive **events** become projections or append-only extensions of the spine; no second brain.
- `docs/superpowers/specs/2026-05-02-trace-bundle-control-plane-design.md` — **TraceBundle** siblings (`signal_snapshot_v1`, `gate_decision_v1`, `decision_trace_v1`) linked by the same `trace_id`; policy gates **never** embedded in the spine.

---

## 1. Purpose

Define one **canonical decision record** (the **Audit Spine**) that:

- Flows through the **entire** pipeline.
- Is the **only** source of truth for:
  - routing decisions,
  - disagreement signals,
  - compression outcomes,
  - adaptive signals (as **derived summaries + proposed updates**, not parallel logs).
- Powers:
  - **UX traces** (human-readable, derived),
  - **CI/benchmark replay** (machine-verifiable, full spine),
  - **Adaptive feedback** (learning input over time; **disabled or read-only** in CI).

---

## 2. Core principle

**No stage recomputes reasoning.**

Every stage **reads** prior spine sections and **appends** its own fields (or nested objects). Earlier stage payloads are **immutable** after write.

**Mindset shift**

- Before: the system decides things and logs them.
- After: the system **is** the evolving record of its own decisions; logs and UX copy are **views** over that record.

---

## 3. Audit Spine schema (v0.1)

Schema version field is mandatory at root (e.g. `schema_version: "audit_spine_v0_1"`). Types below are logical; implementation may use Pydantic models, JSON Schema, or equivalent.

```json
{
  "schema_version": "audit_spine_v0_1",
  "trace_id": "uuid",
  "timestamp": "iso8601",

  "query": {
    "raw": "string",
    "normalized": "string",
    "features": {
      "length": 0,
      "type": "simple|tool|complex"
    }
  },

  "retrieval": {
    "candidates": [
      {
        "doc_id": "string",
        "path": "string",
        "chunk_id": "string",
        "source": "fts|vector|hybrid",
        "score": 0.0,
        "rank": 0
      }
    ],
    "disagreement": {
      "fts_vector_overlap": 0.0,
      "rank_correlation": 0.0,
      "low_overlap_flag": false,
      "mismatch_flag": false
    }
  },

  "routing": {
    "selected_tier": "simple|tool|complex",
    "thresholds": {
      "confidence": 0.0,
      "risk": 0.0
    },
    "risk_flags": ["low_overlap", "high_disagreement"],
    "decision_reason": "string"
  },

  "compression": {
    "input_tokens": 0,
    "output_tokens": 0,
    "retained_chunks": ["chunk_id"],
    "dropped_chunks": ["chunk_id"],
    "loss_metrics": {
      "importance_retained_ratio": 0.0,
      "salient_span_checksum": "string"
    }
  },

  "outcome": {
    "latency_ms": 0,
    "final_confidence": 0.0
  },

  "adaptive": {
    "signals": {
      "retrieval_usefulness": null,
      "compression_loss": null
    },
    "updates": [
      {
        "parameter": "string",
        "delta": 0.0,
        "reason": "string"
      }
    ]
  }
}
```

### 3.1 Privacy and redaction

Default deployment policy should align with adaptive feedback redaction:

- Prefer **hashes / normalized keys** over storing full `query.raw` on disk; bench/CI fixtures may use synthetic or opted-in raw strings.
- Any human-readable UX trace must respect the same redaction profile.

---

## 4. Stage contracts (non-negotiable)

### 4.1 Retrieval stage

| | |
|--|--|
| **Reads** | `query` (and any prior root metadata). |
| **Writes** | `retrieval.candidates`, `retrieval.disagreement`. |
| **Rules** | Disagreement is computed **once** here; **must not** be recomputed in rerank or routing. |

### 4.2 Routing stage

| | |
|--|--|
| **Reads** | `retrieval.*` (and `query.features` as needed). |
| **Writes** | `routing.*`. |
| **Rules** | Decisions **must** reference disagreement flags and **thresholds used** in spine fields; no hidden heuristics outside the spine. |

### 4.3 Compression stage

| | |
|--|--|
| **Reads** | `routing`, `retrieval` (and selected chunk payloads as implied by ids). |
| **Writes** | `compression.*`. |
| **Rules** | Must explicitly record **what** was dropped and **loss metrics**; no silent pruning unreflected in spine. |

### 4.4 Adaptive layer

| | |
|--|--|
| **Reads** | Entire spine (or finalized prefix through `outcome`). |
| **Writes** | `adaptive.signals`, `adaptive.updates` only. |
| **Rules** | **No** direct mutation of routing thresholds or compressor knobs inline; only **proposed** updates with `parameter`, `delta`, and **reason citing spine fields**. Application of updates remains a separate, bounded, reversible mechanism (see adaptive feedback spec). |

---

## 5. Replayability contract

A valid spine must support:

`replay(trace_id)` → **identical decision path** (given frozen inputs).

Requirements:

- **Deterministic** scoring inputs in CI mode (pinned embeddings model off or stubbed per harness spec).
- **Versioned** `schema_version` and tooling version stamps where needed.
- **Stable thresholds** in CI mode (no adaptive writes; same threshold snapshot).

Compatibility note: existing **`decision_trace`** digests should hash **normalized spine subsets** or stable projections so benchmarks do not fork a second metric graph.

---

## 6. Consumers

### 6.1 UX trace (human view)

Derived **only** from the spine:

- Short explanation, key decisions, flagged risks.

Example narrative shape:

- Low overlap detected → widened candidate pool (must appear as routing/compression actions **backed by spine flags**, not prose-only).

### 6.2 Benchmark / CI harness

Consumes **full** spine JSON:

- Compare candidates, rankings, compression ratios, disagreement flags.
- Detect regressions against golden spines or normalized hashes.

**CI must disable adaptive influence** (no applied updates; `adaptive.updates` may be empty or recorded but not executed).

### 6.3 Adaptive engine

Consumes **many** spines over time; emits **bounded** parameter proposals recorded under `adaptive.updates`.

---

## 7. Hard constraints (9.5+ bar)

1. **Single source of truth** — no duplicate disagreement logic; no recomputed signals for the same semantic quantity.
2. **Append-only evolution** — stages add fields; they **never** mutate previously written stage sections.
3. **Deterministic core path** — same inputs → same spine in CI mode.
4. **Bounded adaptation** — rate-limited, reversible, proposals logged in spine (application path explicit and off in CI).

---

## 8. Anti-patterns (kill early)

- Separate ad-hoc logs for routing vs compression vs adaptive that **do not** map to spine fields.
- Recomputing disagreement in reranker or downstream stages.
- Adaptive directly mutating thresholds on the hot path without spine-recorded proposals.
- “Magic” decisions not reflected in the spine.
- CI using **different** logic than runtime (forked scoring paths).

---

## 9. Migration plan (from current code)

| Phase | Action |
|-------|--------|
| **1** | Introduce **AuditSpine** (or equivalent) and pass it through orchestrator/pipeline entrypoints. |
| **2** | Move existing signals into spine sections: e.g. token intelligence → `compression.loss_metrics`; execution route → `routing`; hybrid retrieval → `retrieval` + **single** `disagreement` block. |
| **3** | Remove recomputation; enforce read-from-spine in downstream stages. |
| **4** | Plug CI/benchmark into spine JSON; **no separate parallel metric computation** for merge-blocking assertions. |

---

## 10. Open decisions (v0.2)

- Exact mapping from spine `trace_id` to append-only **adaptive event log** rows (`event_id`, cohort keys).
- Maximum spine size / truncation policy for `candidates` in production vs bench-full mode.
- Whether `routing.thresholds` snapshots a **named profile** id for replay clarity.

---

## Document history

| Rev | Notes |
|-----|--------|
| 0.1 | Initial schema, stage contracts, replay and consumer model, migration phases. |
