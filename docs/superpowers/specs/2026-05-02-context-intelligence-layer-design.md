# Context Intelligence Layer — Design Spec

**Status:** Approved for implementation planning (rev 2 — decision phases, versioning, precedence, telemetry, idempotency)  
**Date:** 2026-05-02  
**Scope:** token-reducer pipeline — decision plane steering retrieval execution, compression, and ranking (not replacing hybrid retrieval).

---

## 1. Purpose

Provide **one control plane** that answers per query:

- What **task** is this (family + behavioral shape)?
- How should we **compress** (delta on existing levels/thresholds)?
- How should we **rank** context (preset or explicit weights)?

**Non-goals:** Replacing FTS/vector retrieval, inventing a parallel RAG stack, or letting an LLM directly set execution tier or telemetry.

**Philosophy:** The layer is a **brain**, not another worker. Downstream stages remain **execution** (retrieve → merge → rerank → subagents → compress → package).

**Architecture class:** **B** — local-first heuristics with **optional LLM fallback** when signals indicate insufficient confidence.

**Product class:** This is an **adaptive context intelligence system with bounded autonomy**, not merely “a token reducer with smart heuristics.”

---

## 2. Decision phase split (critical): D₀ vs D₁

Decisions are **explicitly two-phase** so speculative signals never mix with post-retrieval evidence. This improves **determinism** and **debuggability**.

### 2.1 D₀ — Pre-retrieval

**Inputs:** query text, heuristic intent classification, `execution_route` / `routing_plan` (existing), structured intent defaults (compression level suggestion), query-shape features (length, paths, entropy proxy).

**Produces:** provisional `ContextDecision` fields that do **not** depend on corpus hits:

- `task_family`, `task_shape` (initial)
- Initial `compression_delta` suggestion (may be revised in D₁)
- Initial `ranking_profile` / no weights yet unless heuristic-only
- Preliminary `risk_level` (without weak-retrieval signal)
- **`confidence_D0`** (optional subfield or debug-only; final `confidence` is D₁)

**Does not use:** FTS/vector hit counts, score spread across retrieved chunks, `weak_retrieval`.

### 2.2 D₁ — Post-retrieval refinement

**Inputs:** D₀ decision + **retrieval summary** (`fts_hits`, `vector_hits`, top-K score spread, weak pool flag), merge-stage caps already chosen (may be nudged only within §5.4 bounds).

**Produces:** **final** frozen `ContextDecision`:

- Refined `confidence` (heuristic, control signal)
- `weak_retrieval`, top-3 / pool **score spread** signals
- Final `risk_level` (may move up when retrieval is weak)
- Optional LLM merge (still keyed off D₁ confidence gates)
- **`decision_id`** computed (see §2.5)

**Optional LLM fallback** runs only against **D₁** state (never on speculative D₀ alone).

### 2.3 Debug contract

Telemetry MUST record **phase** (`d0` / `d1`) for snapshots where applicable, or carry separate `signals_d0` vs `signals_d1` under the stable schema (§7).

---

## 3. Core artifact: `ContextDecision`

Structured contract from intelligence → pipeline. All fields must be JSON-serializable for debug and optional caching.

### 3.1 Versioning and identity

| Field | Role |
|-------|------|
| `decision_version` | Schema version string, e.g. **`"v1"`**. Bumped when mandatory fields or merge semantics change (cache invalidation across versions). |
| `decision_id` | **`hash(fallback_cache_key + decision_version)`** (or equivalent stable digest). Used for telemetry correlation, reproducibility, and cache correctness audits. |

### 3.2 Task understanding (two layers)

| Field | Role |
|-------|------|
| `task_family` | Coarse routing: aligns with existing buckets (maps to `legacy_intent` / `StructuredIntent.type` where possible). |
| `task_shape` | Behavioral mode: `single_shot`, `iterative_refinement`, `comparison`, `generation`, `analysis`, etc. |

### 3.3 Compression steering

| Field | Role |
|-------|------|
| `compression_delta` | Numeric delta (e.g. float in `[-1.0, +1.0]` or stepped `-1/0/+1`) applied **after** base route/compression resolution (see consumer precedence §5.3). |

### 3.4 Ranking steering

| Field | Role |
|-------|------|
| `ranking_profile` | Named preset (`semantic_heavy`, `lexical_heavy`, `default`, …). |
| `ranking_weights` | Optional explicit weights; subject to strict validation (§3.5). |

**Precedence (enforced):** See §5.3 — validated `ranking_weights` beat profile; else profile; else system default.

### 3.5 Strict validation for `ranking_weights`

- Sum ≈ `1.0` within tolerance `ε` (e.g. `0.02`).
- No negative values.
- Per-dimension cap (e.g. each weight ≤ `0.7`).
- On any violation → **discard weights entirely**; fall back to `ranking_profile` then system default.

### 3.6 Confidence (operational)

| Field | Role |
|-------|------|
| `confidence` | **Heuristic only**, computed at **D₁** for gating. `[0, 1]`. |
| `confidence_subscores` | Optional: `task`, `strategy`, `ranking`, `disagreement`. |
| `confidence_threshold_used` | Threshold applied this run (config/env). |
| `llm_confidence` | Optional telemetry only; **never** used for gating in v1. |

### 3.7 Risk

| Field | Role |
|-------|------|
| `risk_level` | `low \| medium \| high` — ambiguity, disagreement, weak retrieval, score spread. |

### 3.8 Provenance and fallback metadata

| Field | Role |
|-------|------|
| `provenance` | `heuristic \| llm_fallback \| blended`. |
| `fallback_triggered` | Classifier invoked after cache miss. |
| `fallback_reason` | `low_confidence`, `conflicting_signals`, `weak_retrieval_gated`, `high_risk_policy`, `llm_error`, `timeout`, etc. |

### 3.9 LLM merge metadata

| Field | Role |
|-------|------|
| `override_strength` | `weak \| medium \| strong` — drives dampening vs heuristic (within clamps). |

---

## 4. Confidence and disagreement (heuristic inputs)

**D₁ `confidence`** incorporates at minimum:

- Intent score spread (from `intent.py`).
- Query signals (length, entropy proxy, paths/symbols).
- **Decision disagreement:** mismatch among intent, route tier implications, compression posture, and `context_strategy`.
- **Weak retrieval** + **top-K score spread** (post-retrieval).

Record `confidence_threshold_used`, `fallback_triggered`, `fallback_reason` on every run.

---

## 5. Optional LLM fallback (Architecture B)

### 5.1 Trigger logic (multi-signal)

Evaluate on **D₁** heuristic decision:

- `confidence < confidence_threshold_used`, **or**
- Config: `risk_level == high` forces reconsideration, **or**
- `decision_disagreement` above cutover, **or**
- **`weak_retrieval` gated:** `weak_retrieval == true` **and** `confidence < threshold + margin`.

Triggers are individually toggleable.

### 5.2 Cache / dedup / idempotency

- **`fallback_cache_key`:** hash(normalized query + stable **D₁** signal summary).
- **`decision_id`:** `hash(fallback_cache_key + decision_version)` as in §3.1.
- **TTL:** configurable (e.g. 5–30 minutes).
- **Cache hit:** return **same frozen D₁** `ContextDecision` snapshot (validated once).

**Idempotency requirement:** For fixed **query + D₁ signal vector + config + `decision_version`**, the deterministic heuristic path MUST yield the **same** merged decision (before any intentional randomness). Same inputs → same decision — required for debugging and cache correctness.

### 5.3 Consumer precedence (enforced hierarchy)

**Ranking (apply first match in order):**

1. **`ranking_weights`** — only if validation passes (§3.5).
2. **`ranking_profile`** — preset mapping in `rerank_chunks`.
3. **System default** — current hardcoded blend when neither applies.

**Compression:**

1. **Base level** — execution route + structured intent `compression_level` + reducer threshold logic (existing).
2. **`compression_delta`** — applied on top, clamped.
3. **`risk_level` clamps** — high risk caps aggressive movement (risk-aware merge with LLM per §5.6).

No silent overrides: order is fixed and documented in code comments adjacent to application sites.

### 5.4 Strategy nudging — hard bounds

When `task_shape` / `task_family` **nudge** `context_strategy` (`merge_cap`, `prune_k`, skips):

- **`merge_cap`:** at most **±20%** relative to the value `map_query_to_strategy` would have produced **without** nudge (floor/ceil to sane ints).
- **`prune_k`:** at most **±10%** relative to baseline (same rounding rules).

Prevents the intelligence layer from indirectly overriding routing/tier intent.

### 5.5 Retrieval retry — evidence binding

**Retry is allowed only if:**

- `risk_level == high`, **and**
- **`weak_retrieval == true` OR low top-K score spread** (implementation defines numeric spread threshold consistent with `_weak_scored_pool`-style signals).

Otherwise **no** retrieval retry triggered by this layer (existing policy flags remain).

### 5.6 Classifier input / output / merge

**Input bundle:** bounded JSON-safe payload — trimmed `query`, D₁ decision snapshot, retrieval summary (`fts_hits`, `vector_hits`, score spread, weak pool), numeric disagreement only (no raw corpus dumps).

**Allowlist output:** `task_family`, `task_shape`, `compression_delta`, `ranking_profile` **and/or** `ranking_weights`, `risk_level`, optional `llm_confidence`.

**Explicit rule — both ranking fields from LLM:**

If both **`ranking_profile`** and **`ranking_weights`** are present → apply **`ranking_weights` only if validation succeeds**; if validation fails → use **`ranking_profile`**; if profile missing → system default. Never apply invalid weights.

**Merge:** advisory LLM; risk-aware `compression_delta` blend when heuristic `risk_level == high`; `override_strength` dampening.

**Failures:** timeout / parse / schema → heuristic path only; `fallback_reason` set.

**Timeout:** v1 synchronous; v2 optional async cache populate.

**Pluggability:** NoOp default; host registers classifier via env/DI.

---

## 6. Integration (pipeline)

### 6.1 Lifecycle per query

1. **Build D₀** provisional decision (intent, route, initial task/compression/ranking hints — no corpus signals).
2. Run retrieval through merge sufficient to compute **D₁ signals**.
3. **Build D₁** heuristic decision + `confidence` + final `risk_level` + `decision_id`.
4. **Cache lookup** by `fallback_cache_key` (+ version) → on hit, attach frozen D₁ decision.
5. If gates fire → classifier → validate → merge → update provenance.
6. **Freeze final D₁ `ContextDecision`** on `ContextRunState` (`context_decision`).

### 6.2 Consumers (steering)

| Stage / module | Behavior |
|----------------|----------|
| `context_strategy` | Nudge within §5.4 bounds from `task_shape` / `task_family`. |
| `rerank_chunks` | Apply precedence §5.3. |
| Compressor + route | Apply §5.3 compression hierarchy + risk clamps. |
| Retrieval retry | Only when §5.5 holds. |
| Telemetry | Emit §7 payload when debug enabled. |

### 6.3 Non-goals (v1)

- LLM does not set `execution_route.tier` directly.
- LLM does not mutate heuristic `confidence` used for gating.

---

## 7. Telemetry schema (stable debug payload)

Required when context-intelligence debug is on (exact env flag in implementation plan). Structure:

```json
{
  "decision_id": "<hash>",
  "decision_version": "v1",
  "phase": "d1",
  "provenance": "heuristic | llm_fallback | blended",
  "fallback_triggered": false,
  "fallback_reason": null,
  "confidence": 0.63,
  "confidence_threshold_used": 0.55,
  "risk_level": "medium",
  "override_strength": "medium",
  "signals": {
    "intent_spread": 0.0,
    "disagreement": 0.0,
    "weak_retrieval": false,
    "top3_spread": 0.0,
    "fts_hits": 0,
    "vector_hits": 0
  },
  "cache": {
    "hit": false,
    "fallback_cache_key_prefix": "<optional short prefix for logs>"
  }
}
```

Numeric fields may be rounded for logs; full precision retained in structured debug when needed. This schema is **versioned with `decision_version`** — evolve additively in v2.

---

## 8. Testing strategy

- **Unit:** gate algebra; cache key stability; TTL; weight validation; merge under `risk_level == high`; allowlist stripping; `override_strength`.
- **Precedence:** ranking weights vs profile vs default; compression base → delta → risk clamps.
- **Clamp stress:** strategy nudge ±20% / ±10% boundaries; `compression_delta` clamps.
- **Idempotency:** same query + same D₁ signals + config → identical `decision_id` and equivalent merged fields (deterministic path).
- **Integration:** mock classifier malformed → heuristic unchanged; valid patch → allowlist only.
- **Regression:** classifier disabled → behavior matches pre-feature baselines.

---

## 9. Self-review checklist

| Check | Result |
|-------|--------|
| D₀ / D₁ separation | Explicit; LLM only on D₁. |
| Versioning | `decision_version` + `decision_id` defined. |
| Precedence | Ranking and compression hierarchies fixed. |
| Strategy bounds | ±20% merge_cap, ±10% prune_k. |
| Retry | Bound to high risk + (weak retrieval OR low spread). |
| Telemetry | Stable JSON schema §7. |
| Idempotency | Same inputs → same decision (deterministic path). |
| ranking_profile + ranking_weights | Weights if valid; else profile; else default (§5.6). |

---

## 10. Next step

Implementation plan via **writing-plans** workflow (no code until plan is approved).
