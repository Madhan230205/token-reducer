# Adaptive Workspace Feedback Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Revision 2 of `docs/superpowers/specs/2026-05-02-adaptive-feedback-loop-design.md`: attribution-weighted EMA scoring, staging vs committed actuator state, debounced flush, plugin-local events with optional hook source weights, cold-start and CI guards — composed with existing `feedback_loop_adjustments` / prune EMA without breaking cache keys or tier rules.

**Architecture:** New package `scripts/token_reducer/adaptive_feedback/` holds models, attribution table, append-only event log, in-memory+durable staging EMA, committed snapshot JSON read only by pipeline helpers. `feedback.py` gains thin wrappers that merge **legacy JSONL-derived** adjustments with **committed adaptive** deltas (multiplicative/additive composition documented below). `pipeline.py` / `orchestrator.py` append normalized outcome events after successful runs and call `maybe_schedule_flush()` (non-blocking). Context intelligence caps apply in `Guardrails` before promotion. Benchmark/CI: `TOKEN_REDUCER_ADAPT_DISABLE=1` skips promotion and adaptive merge.

**Tech stack:** Python 3.11+, stdlib (`json`, `hashlib`, `pathlib`, `dataclasses`, `enum`), existing `pytest`, `ruff`, `mypy`. Cross-read `docs/superpowers/specs/2026-05-02-context-intelligence-layer-design.md` for clamp constants you must not exceed when touching strategy/rerank paths.

**Specs:** Primary `docs/superpowers/specs/2026-05-02-adaptive-feedback-loop-design.md`; compatibility `docs/superpowers/specs/2026-05-02-benchmark-proof-harness-design.md`, `docs/superpowers/specs/2026-05-02-context-intelligence-layer-design.md`.

---

## File map (create / modify)

| Path | Role |
|------|------|
| Create `scripts/token_reducer/adaptive_feedback/__init__.py` | Public exports |
| Create `scripts/token_reducer/adaptive_feedback/models.py` | Enums, dataclasses, schema version |
| Create `scripts/token_reducer/adaptive_feedback/constants.py` | Defaults T=10, M=25, weights, env parsing |
| Create `scripts/token_reducer/adaptive_feedback/attribution.py` | Normative signal→actuator table + `ambiguous` flags |
| Create `scripts/token_reducer/adaptive_feedback/redaction.py` | Hash prompts, bound diagnostic strings |
| Create `scripts/token_reducer/adaptive_feedback/event_log.py` | Append-only JSONL, malformed skip counter |
| Create `scripts/token_reducer/adaptive_feedback/scorer.py` | Weighted EMA increments into `StagingState` |
| Create `scripts/token_reducer/adaptive_feedback/actuators.py` | Map staged scores → bounded knob deltas |
| Create `scripts/token_reducer/adaptive_feedback/state_store.py` | Paths, load/save staging & committed, atomic promote, `.bak` |
| Create `scripts/token_reducer/adaptive_feedback/debouncer.py` | Time + mass gates, `maybe_schedule_flush` |
| Create `scripts/token_reducer/adaptive_feedback/guardrails.py` | Daily caps, min_samples gate, anomaly stub (hook for harness later) |
| Create `scripts/token_reducer/adaptive_feedback/collector.py` | Build `OutcomeEvent` from `ContextRunState` / hook dict |
| Modify `scripts/token_reducer/feedback.py` | `feedback_loop_adjustments_with_adaptive()`, document merge order |
| Modify `scripts/token_reducer/pipeline.py` | Use merged adjustments; optional adapt event append after `log_result` |
| Modify `scripts/token_reducer/orchestrator.py` | Same merge where `feedback_loop_adjustments()` is called |
| Modify `scripts/token_reducer/multi_agent/agents.py` | Same merge if it constructs adjustments independently |
| Tests `tests/test_adaptive_feedback_*.py` | Unit + integration per sections below |

---

### Task 1: Package skeleton and models

**Files:**
- Create: `scripts/token_reducer/adaptive_feedback/__init__.py`
- Create: `scripts/token_reducer/adaptive_feedback/models.py`
- Create: `scripts/token_reducer/adaptive_feedback/constants.py`
- Test: `tests/test_adaptive_feedback_models.py`

- [ ] **Step 1: Write failing test for schema version and enum membership**

```python
# tests/test_adaptive_feedback_models.py
from token_reducer.adaptive_feedback.models import SCHEMA_VERSION, SignalType


def test_schema_version_fixed() -> None:
    assert SCHEMA_VERSION == "adapt_feedback_v1"


def test_signal_type_contains_taxonomy() -> None:
    names = {s.value for s in SignalType}
    for x in (
        "retrieval_hit_strong",
        "retrieval_miss_weak_pool",
        "compression_adequate",
        "session_flow_smooth",
        "follow_up_tightening",
        "hook_tool_failure",
        "baseline_tick",
    ):
        assert x in names
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `pytest tests/test_adaptive_feedback_models.py -v`  
Expected: `ModuleNotFoundError` / import failure.

- [ ] **Step 3: Implement minimal models + constants**

```python
# scripts/token_reducer/adaptive_feedback/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

SCHEMA_VERSION = "adapt_feedback_v1"


class SignalType(str, Enum):
    RETRIEVAL_HIT_STRONG = "retrieval_hit_strong"
    RETRIEVAL_MISS_WEAK_POOL = "retrieval_miss_weak_pool"
    COMPRESSION_ADEQUATE = "compression_adequate"
    SESSION_FLOW_SMOOTH = "session_flow_smooth"
    FOLLOW_UP_TIGHTENING = "follow_up_tightening"
    HOOK_TOOL_FAILURE = "hook_tool_failure"
    BASELINE_TICK = "baseline_tick"


SourceKind = Literal["local", "hook"]


@dataclass(frozen=True)
class OutcomeEvent:
    schema_version: str
    event_id: str
    ts_epoch: float
    workspace_fingerprint: str
    source: SourceKind
    signal_type: SignalType
    magnitude: float
    cohort_key: tuple[str, ...]
    correlation: dict[str, Any] | None = None


@dataclass
class StagingState:
    """EMA nets per cohort × actuator channel — implementation narrows fields in Task 4."""
    cohort_utility: dict[tuple[str, ...], float] = field(default_factory=dict)
    cohort_penalty: dict[tuple[str, ...], float] = field(default_factory=dict)
    samples_per_cohort: dict[tuple[str, ...], int] = field(default_factory=dict)


@dataclass
class CommittedActuators:
    """Knobs the pipeline may read — v1 numeric biases only."""
    retrieval_scale_mult_delta: float = 0.0
    relevance_floor_delta: float = 0.0
    prune_bias_ema_delta: dict[str, float] = field(default_factory=dict)
    skill_prior_delta: dict[str, float] = field(default_factory=dict)
```

```python
# scripts/token_reducer/adaptive_feedback/constants.py
from __future__ import annotations

import os

DEFAULT_FLUSH_INTERVAL_MINUTES = 10
DEFAULT_FLUSH_EVENT_BATCH = 25
DEFAULT_MIN_SAMPLES = 8
DEFAULT_HOOK_WEIGHT = 1.75
DEFAULT_HOOK_WEIGHT_MAX = 2.0
DEFAULT_LOCAL_WEIGHT = 1.0


def env_positive_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def flush_interval_minutes() -> int:
    return env_positive_int("TOKEN_REDUCER_ADAPT_FLUSH_INTERVAL_MINUTES", DEFAULT_FLUSH_INTERVAL_MINUTES)


def flush_event_batch() -> int:
    return env_positive_int("TOKEN_REDUCER_ADAPT_FLUSH_EVENT_BATCH", DEFAULT_FLUSH_EVENT_BATCH)


def min_samples_actuation() -> int:
    return env_positive_int("TOKEN_REDUCER_ADAPT_MIN_SAMPLES", DEFAULT_MIN_SAMPLES)


def hook_source_weight() -> float:
    raw = (os.environ.get("TOKEN_REDUCER_ADAPT_HOOK_WEIGHT") or "").strip()
    if not raw:
        w = DEFAULT_HOOK_WEIGHT
    else:
        try:
            w = float(raw)
        except ValueError:
            w = DEFAULT_HOOK_WEIGHT
    return max(1.0, min(DEFAULT_HOOK_WEIGHT_MAX, w))


def local_source_weight() -> float:
    return DEFAULT_LOCAL_WEIGHT


def adaptive_disabled() -> bool:
    if (os.environ.get("TOKEN_REDUCER_ADAPT_DISABLE") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return True
    return False
```

```python
# scripts/token_reducer/adaptive_feedback/__init__.py
from .constants import adaptive_disabled
from .models import SCHEMA_VERSION, CommittedActuators, OutcomeEvent, SignalType, StagingState

__all__ = [
    "SCHEMA_VERSION",
    "SignalType",
    "OutcomeEvent",
    "StagingState",
    "CommittedActuators",
    "adaptive_disabled",
]
```

- [ ] **Step 4: Run tests — PASS**

Run: `pytest tests/test_adaptive_feedback_models.py -v`

- [ ] **Step 5: Commit**

```bash
git add scripts/token_reducer/adaptive_feedback tests/test_adaptive_feedback_models.py
git commit -m "feat(adaptive): scaffold models and constants"
```

---

### Task 2: Attribution table and tests

**Files:**
- Create: `scripts/token_reducer/adaptive_feedback/attribution.py`
- Test: `tests/test_adaptive_feedback_attribution.py`

- [ ] **Step 1: Test — every SignalType has targets and ambiguity flag**

```python
# tests/test_adaptive_feedback_attribution.py
from token_reducer.adaptive_feedback.attribution import ACTUATOR_FAMILIES, attribution_for
from token_reducer.adaptive_feedback.models import SignalType


def test_each_signal_has_row() -> None:
    for st in SignalType:
        row = attribution_for(st)
        assert row.targets <= ACTUATOR_FAMILIES
        assert isinstance(row.ambiguous, bool)


def test_baseline_tick_targets_empty() -> None:
    assert attribution_for(SignalType.BASELINE_TICK).targets == frozenset()
```

- [ ] **Step 2: Implement `attribution.py`**

Use frozen dataclass `AttributionRow(targets: frozenset[str], ambiguous: bool)`. Encode families as `"feedback_multipliers"`, `"strategy_prune"`, `"skill_priors"` matching spec Section 6 list (omit context_intel as separate channel — applied via guardrails). Populate rows per spec Section 6.1 table; mark `follow_up_tightening` `ambiguous=True`.

- [ ] **Step 3: pytest PASS + commit**

---

### Task 3: Event log (append + tolerant read)

**Files:**
- Create: `scripts/token_reducer/adaptive_feedback/event_log.py`
- Create: `scripts/token_reducer/adaptive_feedback/redaction.py`
- Test: `tests/test_adaptive_feedback_event_log.py`

- [ ] **Step 1: Implement `_default_adapt_dir()`** parallel to `feedback._default_log_path()` parent: same root `.cache/token-reducer/adaptive/` (or under `CLAUDE_PLUGIN_ROOT`).

- [ ] **Step 2: `append_event(path, OutcomeEvent)`** serializes dict with `signal_type` as string; never raises to caller (catch `OSError`).

- [ ] **Step 3: `iter_recent_events(path, tail_lines)`** skips bad JSON lines, returns list.

- [ ] **Step 4: Tests** write tmp JSONL with one bad line + two good; assert len good == 2.

---

### Task 4: Scorer — weighted EMA into staging

**Files:**
- Create: `scripts/token_reducer/adaptive_feedback/scorer.py`
- Test: `tests/test_adaptive_feedback_scorer.py`

- [ ] **Step 1: Implement `apply_event(state: StagingState, event: OutcomeEvent) -> None`**

Use `source_weight`: `hook` → `hook_source_weight()`, `local` → `local_source_weight()`. Map signal polarity to utility vs penalty increment (simple v1: positive signals add utility, negative add penalty). Increment `samples_per_cohort[event.cohort_key]`. Use attribution to skip actuator-irrelevant signals for scoring channels (baseline_tick only decay — implement decay helper reducing magnitudes slightly when optional).

- [ ] **Step 2: Test hook vs local** same magnitude → staging utility differs by weight ratio.

---

### Task 5: Actuators + guardrails + state store

**Files:**
- Create: `scripts/token_reducer/adaptive_feedback/actuators.py`
- Create: `scripts/token_reducer/adaptive_feedback/guardrails.py`
- Create: `scripts/token_reducer/adaptive_feedback/state_store.py`
- Test: `tests/test_adaptive_feedback_state_store.py`

- [ ] **Step 1: `staging_to_committed(staging, *, min_samples)`** returns `CommittedActuators` only if every touched cohort in staging has `samples >= min_samples`; otherwise returns **neutral** committed (all zeros / empty dicts) per Section 5.2.

- [ ] **Step 2: `guardrails.clamp_committed(c: CommittedActuators)`** enforce global max deltas (pick conservative constants in code, e.g. `|retrieval_scale_mult_delta| <= 0.06` applied as additive to mult composition — document in module docstring).

- [ ] **Step 3: `state_store.promote_staging_atomic(workspace_root, staging, committed)`** write `staging.json.tmp` → rename; same for `committed.json`; copy previous committed to `committed.json.bak`.

- [ ] **Step 4: Tests** tmp_path: promote twice, corrupt committed mid-write (simulate), loader restores `.bak`.

---

### Task 6: Debouncer

**Files:**
- Create: `scripts/token_reducer/adaptive_feedback/debouncer.py`
- Test: `tests/test_adaptive_feedback_debouncer.py`

- [ ] **Step 1: Track `last_flush_ts`, `events_since_flush` in small JSON `debouncer_meta.json` beside adaptive dir.

- [ ] **Step 2: `record_event_and_should_flush()`** returns bool when time OR mass threshold hit.

- [ ] **Step 3: `maybe_schedule_flush`** runs synchronously in v1 (simple): if disabled via `adaptive_disabled()`, return immediately; else if should flush, run promote pipeline (load staging from disk or memory — choose one pattern and stick to it).

---

### Task 7: Collector from pipeline state

**Files:**
- Create: `scripts/token_reducer/adaptive_feedback/collector.py`
- Test: `tests/test_adaptive_feedback_collector.py`

- [ ] **Step 1: `cohort_key_from_state(state: ContextRunState) -> tuple[str, ...]`** using `execution_route.tier`, `context_strategy.strategy_id`, `route.skill_id`, coarse intent bucket from `state.intent` (hash or enum — stable stringify).

- [ ] **Step 2: Derive at least two local signals from existing fields** e.g. weak pool / retry flags from `state` if present; else emit `baseline_tick` low magnitude. Document mapping in module docstring per spec Section 4.

- [ ] **Step 3: `build_events_from_run(state) -> list[OutcomeEvent]`** stable `event_id` from hash of (ts, cohort, signal, correlation digest).

---

### Task 8: Merge with `feedback_loop_adjustments`

**Files:**
- Modify: `scripts/token_reducer/feedback.py`
- Test: `tests/test_adaptive_feedback_merge.py`

- [ ] **Step 1: Add `feedback_loop_adjustments_with_adaptive(log_path=None, workspace_root=None)`**

Compute `base = feedback_loop_adjustments(log_path=...)`. If `adaptive_disabled()` or missing committed file, return `base`. Else load `CommittedActuators` and compose:

- **v1 additive composition:** `retrieval_scale_mult = clamp(base.retrieval_scale_mult + adaptive.retrieval_scale_mult_delta)` to `[0.84, 1.09]` (same envelope as `feedback_loop_adjustments` output range).

- `relevance_floor_delta = base.relevance_floor_delta + adaptive.relevance_floor_delta` clamped to `[-0.022, 0.045]`.

Document merge order in docstring: **legacy JSONL first, adaptive committed layered second**.

- [ ] **Step 2: Replace call sites** in `pipeline.py`, `orchestrator.py`, `multi_agent/agents.py` from `feedback_loop_adjustments()` to `feedback_loop_adjustments_with_adaptive(..., workspace_root=workspace_root)` passing through workspace root where available.

- [ ] **Step 3: Run full suite** `pytest` + `ruff check` + `mypy` per `Makefile`/project config.

---

### Task 9: Wire ingestion + flush post-run

**Files:**
- Modify: `scripts/token_reducer/pipeline.py`

- [ ] **Step 1: After `log_result` block** (inside `TOKEN_REDUCER_FEEDBACK`), if not `adaptive_disabled()`, append collector events, update staging file, increment debouncer, call `maybe_schedule_flush`.

- [ ] **Step 2: Ensure cache key material** in `run_retrieval_pipeline` uses merged adjustments (Task 8) so adaptive changes invalidate cache consistently.

---

### Task 10: Strategy prune merge (family 2)

**Files:**
- Modify: `scripts/token_reducer/feedback.py` or `context_strategy.py` consumer

- [ ] **Step 1: Where `read_strategy_prune_adjustments` feeds prune deltas**, add optional tiny bias from `CommittedActuators.prune_bias_ema_delta` clamped per strategy ±1.

---

### Task 11: Skill priors (family 1) — minimal v1

**Files:**
- Modify: subagent selection site (e.g. `subagents/agents.py` or router)

- [ ] **Step 1: Load committed `skill_prior_delta` dict** keyed by agent/skill id; apply as additive score bump capped (e.g. ±0.02) when ranking agents. If no stable id match, no-op.

---

### Task 12: Documentation and self-review

- [ ] **Step 1: Add module-level README section** in plan only is insufficient — add short **comment block** in `adaptive_feedback/__init__.py` pointing to spec path.

- [ ] **Step 2: Spec coverage checklist** — walk every numbered section of `2026-05-02-adaptive-feedback-loop-design.md` and confirm a task maps (fill gaps if any).

- [ ] **Step 3: Final commit** `docs: note adaptive feedback implementation entry` optional if you add developer-facing paragraph to existing doc (only if project already documents env vars elsewhere).

---

## Merge semantics (normative for implementers)

1. **Pipeline read path:** only `CommittedActuators` on disk (or neutral defaults).
2. **Staging** updates on every eligible event; **promotion** runs on debounced flush; until promotion, pipeline sees previous committed.
3. **Cold-start:** `staging_to_committed` returns neutral if `samples_per_cohort[cohort] < min_samples` for any cohort touched since last flush (spec: no committed bias — implement by refusing promotion while staging holds under-sampled cohorts, **or** by promoting neutral snapshot only; choose one and test).

## Self-review (plan author)

| Spec section | Task coverage |
|--------------|---------------|
| 2 Sources A/B | Task 7 collector source field; Task 4 weights |
| 3 Event model | Task 1 models; Task 3 log |
| 4 Taxonomy + 4.1 ambiguity | Task 1–2 |
| 5 Cohort + 5.1 weights + 5.2 cold-start | Task 7; Task 4; Task 5 |
| 6 Actuators + 6.1 attribution | Task 2; Task 5; Task 8–11 |
| 7 Debounce + staging | Task 5–6; Task 9 |
| 8 Safety | Task 5 guardrails (extend for daily caps in same module) |
| 9 Components | File map |
| 10 Testing | Each task tests |

**Placeholder scan:** none intentional; numeric caps in Task 5/8 are starter values — tune once against context-intelligence constants file (`context_intelligence/constants.py`).

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-05-02-adaptive-feedback-loop.md`.

**Deprecation reminder:** `/execute-plan` is deprecated; use **superpowers:executing-plans** with this plan file, or **superpowers:subagent-driven-development** for per-task agents.

**Which execution mode do you want: subagent-driven (recommended) or inline executing-plans?**
