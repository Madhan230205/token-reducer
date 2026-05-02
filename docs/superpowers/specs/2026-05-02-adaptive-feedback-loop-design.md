# Adaptive Workspace Feedback Loop — Design Spec

**Status:** Approved for implementation planning  
**Date:** 2026-05-02  
**Scope:** token-reducer — bounded, debounced **learning** from per-turn outcomes to nudge **allowlisted** pipeline knobs (skill priors, strategy prune EMA, feedback multipliers, context-intelligence-adjacent caps), without neural training or unbounded self-modification.

**Related:** Consumes **decision-trace-shaped** signals compatible with `docs/superpowers/specs/2026-05-02-benchmark-proof-harness-design.md` and must **respect** bounds from `docs/superpowers/specs/2026-05-02-context-intelligence-layer-design.md` where overlap exists.

---

## 1. Purpose

Close the loop between **static heuristics** (routing, skill pick, compression, retrieval policy) and **observed session behavior** so the system **gradually** prefers configurations that correlate with better outcomes for **this workspace**.

**Primary outcomes**

- **Portable baseline:** works with **plugin-local signals only** (required path).
- **Richer truth when available:** optional **host/hook** events improve signal quality without breaking portability.
- **Auditability:** append-only event log + versioned state snapshots; no silent unbounded drift.
- **Safety:** debounced disk writes, atomic files, rollback, explicit disable in benchmark/CI modes.

**Non-goals (v1)**

- Reinforcement learning, model training, or LLM-in-the-loop critics on the hot path.
- Rewriting user prompts, free-form policy mutation, or mutating `execution_route` tier rules directly.
- Parallel speculative subagents, SOTA retrieval rewrite science, or “wow” anticipatory prefetch (separate specs).

---

## 2. Signal sources (precedence)

### 2.1 Required — plugin-local (always)

Sources **must** exist in every supported deployment:

- Pipeline summaries already derivable without network: routing tier, strategy id, retrieval retry fired, pool weakness proxies, selected sources (paths only), token estimates, compression path indicators, optional **`decision_trace`**-compatible snapshot ids (hashes over normalized fields).
- **Session-local proxies** (noisy but universal): repeated queries on similar keys, rapid follow-up queries, same-file focus streaks (derived from session memory and hashed query keys — **not** raw prompt logging by default).

### 2.2 Optional — hooks / host events

When the environment exposes structured hooks:

- Tool failures, generation stops, explicit accept/reject metadata, contradiction follow-ups (schema’d enums only).

**Rule:** absence of hooks **must not** prevent learning from **2.1**. Hooks **add** signal mass and confidence; they **never** define exclusive code paths that skip **2.1**.

### 2.3 Privacy and redaction

- Default: **no full user prompts** in outcome events; store **hashes**, **intent buckets**, and **short diagnostic strings** bounded by length.
- Hook payloads undergo the same redaction profile before append.

---

## 3. Outcome event model

Append-only records (JSON Lines or sqlite sidecar — implementation chooses; schema version **`adapt_feedback_v1`**).

**Minimum fields**

| Field | Role |
|-------|------|
| `schema_version` | e.g. `adapt_feedback_v1` |
| `event_id` | stable ulid or hash |
| `ts_epoch` | float seconds |
| `workspace_fingerprint` | hash of normalized workspace root path + optional repo id |
| `source` | `local` \| `hook` |
| `signal_type` | closed enum (see Section 4) |
| `magnitude` | float in **[0, 1]** or small int severity mapped to float |
| `cohort_key` | normalized tuple (see Section 5) |
| `correlation` | optional ids linking to prior pipeline run (`decision_trace` digest, session id) |

**Errors:** malformed lines are skipped with counter increment; never crash the pipeline.

---

## 4. Signal taxonomy (v1 enums)

Start **small**; extend additively with version bumps.

**Positive-leaning**

- `retrieval_hit_strong` — expected paths satisfied / high recall proxy from deterministic checks when available
- `compression_adequate` — output length within predicted band for intent bucket
- `session_flow_smooth` — no rapid contradiction follow-up within window (proxy only)

**Negative-leaning**

- `retrieval_miss_weak_pool` — weak pool / retry patterns aligned with existing flags
- `follow_up_tightening` — user re-queries with narrower scope soon after (proxy)
- `hook_tool_failure` — from hooks only

**Neutral / meta**

- `baseline_tick` — heartbeat for decay (optional)

Exact mapping from raw pipeline metrics → enum is implementation-defined but **must** be documented beside code.

---

## 5. Cohort keys and scoring

**Cohort key** (examples; implementation may normalize further):

- `(execution_tier, strategy_id, tool_skill_id_or_null, intent_bucket)` — normalized tuple format fixed by implementation

**Scoring**

- Maintain **EMA** for **utility** and **penalty** per cohort (two scalars or one net with separate decay — implementation picks one with tests).
- **EMA half-life** and **min samples before actuation** are required constants (implementation plan proposes defaults; must be tunable via env).
- **Decay:** stale cohorts lose influence automatically.

---

## 6. Actuators (allowlist)

Only the following families may change from learned updates:

1. **Skill registry priors** — efficiency / success proxy weights used in TOOL-tier selection (bounded deltas).
2. **Strategy prune adjustments** — compatible with existing workspace EMA files (`read_strategy_prune_adjustments` pattern).
3. **Feedback loop multipliers** — retrieval scale, relevance floor delta (already present as feedback adjustments), with **hard caps**.
4. **Context intelligence compatibility** — any nudge that overlaps context intelligence **must clamp** to documented bounds (strategy nudge caps, retry policy constraints, ranking precedence).

**Forbidden:** mutating raw tier classification rules, embedding model choice, or bypassing proof-harness disable flags.

---

## 7. Debounced persistence

**Flush triggers (either satisfies a write):**

- **Time:** default **`T = 10` minutes** since last successful flush for this workspace.
- **Mass:** default **`M = 25`** new eligible events since last flush.

**Overrides (env, names illustrative):**

- `TOKEN_REDUCER_ADAPT_FLUSH_INTERVAL_MINUTES` (default `10`)
- `TOKEN_REDUCER_ADAPT_FLUSH_EVENT_BATCH` (default `25`)

**Mechanics**

- **Non-blocking:** scoring may occur inline; **disk commit** happens only on flush.
- **Atomic write:** temp file + rename; maintain **last-known-good** snapshot.
- **Crash recovery:** partial files rejected at startup; restore snapshot or defaults.
- **Benchmark / CI guard:** when benchmark disable env is set (same family as proof harness `TOKEN_REDUCER_BENCHMARK_*` or dedicated `TOKEN_REDUCER_ADAPT_DISABLE=1`), **no actuator writes** and optionally **no event append** (implementation chooses one level of disable; document clearly).

---

## 8. Safety and rollback

| Guard | Requirement |
|-------|-------------|
| Max delta per knob per day | Hard cap per actuator family |
| Min cohort samples | No write until threshold met |
| Anomaly detection | Sudden multi-sigma swing → revert last flush |
| Harness regression | If proof-harness **required** baseline exists and shows regression on cohort overlap, **rollback** last actuator snapshot |

---

## 9. Components

| Unit | Responsibility |
|------|----------------|
| `SignalCollector` | Normalize pipeline + optional hook payloads → events |
| `EventLog` | Append-only store with rotation cap |
| `Scorer` | EMA updates per cohort |
| `ActuatorApplier` | Translate net scores → bounded deltas on allowlist |
| `Debouncer` | Time + mass flush scheduler (per workspace) |
| `Guardrails` | Clamps vs context intelligence + daily caps |

---

## 10. Testing strategy

- Unit: EMA decay, debouncer ordering, enum mapping purity, redaction.
- Property-style: proposed deltas never exceed configured caps.
- Integration: synthetic event stream → expect deterministic actuator file content.
- Regression: with `ADAPT_DISABLE`, filesystem state unchanged across runs.

---

## 11. Self-review checklist

| Check | Result |
|-------|--------|
| Plugin-local required path | Always-on learning possible |
| Hooks optional | Enrichment only |
| Debounced writes | `T=10`, `M=25` defaults |
| Allowlist only | No tier/prompt mutation |
| Intel / harness compatibility | Clamps + CI disable |
| Privacy | No raw prompts by default |

---

## 12. Next step

Use the **writing-plans** workflow for implementation (wire `SignalCollector` after pipeline stages, file locations under ignored workspace cache, CLI inspection commands, migration from any existing feedback logs).
