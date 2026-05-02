# Context Intelligence Layer — Design Spec

**Status:** Approved for implementation planning  
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

---

## 2. Core artifact: `ContextDecision`

Structured contract from intelligence → pipeline. All fields must be JSON-serializable for debug and optional caching.

### 2.1 Task understanding (two layers)

| Field | Role |
|-------|------|
| `task_family` | Coarse routing: aligns with existing buckets (e.g. maps to `legacy_intent` / `StructuredIntent.type` where possible). May extend with `chat \| code \| analysis \| …` consistent with current enums. |
| `task_shape` | Behavioral mode affecting compression/ranking more than category: e.g. `single_shot`, `iterative_refinement`, `comparison`, `generation`, `analysis`. |

### 2.2 Compression steering

| Field | Role |
|-------|------|
| `compression_delta` | **Numeric delta** (e.g. float in `[-1.0, +1.0]` or stepped `-1/0/+1`) applied **on top of** existing `compression_level` and route thresholds — avoids branching explosion of named biases only. |

### 2.3 Ranking steering

| Field | Role |
|-------|------|
| `ranking_profile` | Named preset (e.g. `semantic_heavy`, `lexical_heavy`, `default`) mapping to the existing rerank blend in `rerank_chunks`. |
| `ranking_weights` | Optional explicit weights dict. **Rule:** If `ranking_weights` is present **and** validates → **override preset**. Else use preset. |

**Strict validation for `ranking_weights`:**

- Sum ≈ `1.0` within tolerance `ε` (e.g. `0.02`).
- No negative values.
- Per-dimension cap (e.g. each weight ≤ `0.7`).
- On any violation → **discard weights entirely**, keep `ranking_profile` only.

### 2.4 Confidence (operational)

| Field | Role |
|-------|------|
| `confidence` | **Heuristic only** — single control signal in `[0, 1]`. Used for gates and telemetry. |
| `confidence_subscores` | Optional: `task`, `strategy`, `ranking`, `disagreement` — for tuning only. |
| `confidence_threshold_used` | Threshold applied this run (from config/env) — required for reproducible debugging. |
| `llm_confidence` | Optional; **never used for gating** in v1 — stored for telemetry only. |

**Rule:** Do **not** blend heuristic and LLM confidence for control decisions.

### 2.5 Risk

| Field | Role |
|-------|------|
| `risk_level` | `low \| medium \| high` — derived from ambiguity, vagueness, intent disagreement, weak retrieval. Drives conservative compression bias, fallback likelihood (with gates), retry aggressiveness caps. |

### 2.6 Provenance and fallback metadata

| Field | Role |
|-------|------|
| `provenance` | `heuristic \| llm_fallback \| blended`. |
| `fallback_triggered` | Whether optional classifier was invoked this query (after cache miss). |
| `fallback_reason` | When relevant: `low_confidence`, `conflicting_signals`, `weak_retrieval_gated`, `high_risk_policy`, `llm_error`, `timeout`, etc. |

### 2.7 LLM merge metadata

| Field | Role |
|-------|------|
| `override_strength` | `weak \| medium \| strong` — derived from how many allowlisted fields changed and magnitude vs heuristic. Controls merge dampening: weak → partial blend toward heuristic; strong → full apply within clamps. |

---

## 3. Confidence and disagreement (heuristic inputs)

Heuristic `confidence` incorporates at minimum:

- Spread between top intent classifier scores (`intent.py`).
- Query signals (length, entropy proxy, presence of paths/symbols).
- **Decision disagreement:** mismatch among intent classification, `execution_route` tier implications, chosen compression aggressiveness, and `context_strategy` — inconsistency lowers confidence.
- **Weak retrieval** (after bounded retrieval summary available): flag + pool strength.

**Operational extensions:**

- `confidence_threshold_used` recorded on every decision.
- `fallback_triggered` + `fallback_reason` set whenever the optional path runs or fails.

---

## 4. Optional LLM fallback (Architecture B)

### 4.1 Trigger logic (multi-signal)

Evaluate **after** heuristic `ContextDecision` exists:

- `confidence < confidence_threshold_used`, **or**
- Policy: `risk_level == high` may force reconsideration (configurable), **or**
- `decision_disagreement` above cutover, **or**
- **`weak_retrieval` gated:** `weak_retrieval == true` **and** `confidence < threshold + margin` (margin configurable — prevents noisy LLM calls when heuristic is already confident).

All triggers are **config-toggleable** per signal.

### 4.2 Cache / dedup

- **`fallback_cache_key`:** hash of **normalized query** + **stable numeric signal summary** (intent top scores, weak_retrieval, disagreement scalar — not full corpus).
- **TTL:** configurable (e.g. **5–30 minutes**).
- **On hit:** reuse **frozen post-merge** `ContextDecision` snapshot (validated once) to avoid double-merge and drift.

### 4.3 Classifier input bundle

Bounded JSON-safe payload:

- Trimmed `query` (hard max chars).
- Full heuristic `ContextDecision` minus any secret fields (none expected in v1).
- Retrieval summary: `fts_hits`, `vector_hits`, top-score spread, `weak_pool` boolean.
- Numeric **disagreement summary** only (not raw internal dumps).

### 4.4 Classifier output — strict allowlist

LLM may propose **only**:

- `task_family`
- `task_shape`
- `compression_delta`
- `ranking_profile` **or** `ranking_weights` (not both conflicting — if both sent, define precedence: weights win only if valid)
- `risk_level`
- Optional `llm_confidence` (telemetry)

**Enforcement:** schema validation + strip unknown keys; never trust prompt discipline alone.

### 4.5 Merge rules

- LLM is **advisory**; heuristic telemetry preserved.
- **`compression_delta`:** clamp to allowed range; **if heuristic `risk_level == high`**, apply **risk-aware blend** — do not fully adopt aggressive LLM deltas (cap movement toward aggressive compression).
- **`ranking_weights`:** validate strictly or discard entirely.
- **`override_strength`:** computed from delta magnitude and field count; drives how aggressively LLM patches replace heuristic fields (within clamps).

### 4.6 Failure handling

- Timeout, parse error, schema violation → **keep heuristic decision**, set `fallback_triggered=True`, `fallback_reason` accordingly; **never block** pipeline execution.

### 4.7 Timeout behavior

- **v1:** synchronous call with strict timeout → on exceed, heuristic path only.
- **v2 (optional):** async fire; late result ignored for current request but **may populate cache** for future `fallback_cache_key` hits — document as follow-up.

### 4.8 Pluggability

- Default: **NoOp** classifier (never calls network).
- Host/plugin registers callable or HTTP adapter via env or DI (exact mechanism in implementation plan).

---

## 5. Integration (pipeline)

### 5.1 Lifecycle per query

1. Build **heuristic** `ContextDecision` + `confidence` + `risk_level` + disagreement + threshold used.
2. **Cache lookup** by `fallback_cache_key` → if valid hit, attach decision and skip LLM.
3. If triggers fire → optional classifier → validate → merge with risk-aware rules → set `provenance` / `fallback_*`.
4. **Freeze** final `ContextDecision` on **`ContextRunState`** (e.g. `context_decision`).

### 5.2 Consumers (steering)

| Stage / module | Behavior |
|----------------|----------|
| `context_strategy` / `map_query_to_strategy` | `task_shape` + `task_family` adjust caps, skips, attention framing. |
| `rerank_chunks` | Apply `ranking_weights` or `ranking_profile` preset. |
| Compressor + execution route thresholds | Apply `compression_delta`; `risk_level` biases conservatism. |
| Retrieval retry policy | Bounded by `risk_level` + signals (no second LLM). |
| Debug | Redacted `ContextDecision`, cache hit/miss, reasons — behind existing debug flags. |

### 5.3 Explicit non-goals (v1)

- LLM does **not** set `execution_route.tier` directly.
- LLM does **not** mutate `confidence` used for gating.

---

## 6. Testing strategy

- **Unit:** gate algebra; cache key stability; TTL expiry; weight validation failures; merge under `risk_level == high`; allowlist stripping; override_strength behavior.
- **Integration:** mock classifier returns malformed JSON → heuristic unchanged; valid patch → merged fields only within allowlist.
- **Regression:** existing pipeline tests unchanged behavior when classifier disabled.

---

## 7. Self-review checklist

| Check | Result |
|-------|--------|
| Placeholders / TBD | None intentional; numeric ε/margins/TTL finalized in implementation plan from sensible defaults. |
| Internal consistency | Two-layer task model + delta compression + weights-over-preset hierarchy aligned. |
| Scope | Single coherent feature; async timeout deferred to v2. |
| Ambiguity | `ranking_profile` vs `ranking_weights`: weights override only when valid; if LLM sends both, implementation plan must pick one rule (recommend: prefer validated weights, else profile). |

---

## 8. Next step

Implementation plan via **writing-plans** workflow (no code until plan is approved).
