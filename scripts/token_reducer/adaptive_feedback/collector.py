"""Build :class:`~token_reducer.orchestrator.ContextRunState` → outcome events (plan Task 7).

Local signal mapping (v1):
- Weak scored pool (same heuristic as orchestrator retrieval) → ``retrieval_miss_weak_pool``.
- Retrieval retry fired → extra tightening proxy → ``follow_up_tightening`` (bounded magnitude).
- Otherwise healthy top score → ``retrieval_hit_strong``.
- Fallback → ``baseline_tick`` for EMA decay driver.

Cohort tuple: ``(execution_tier, strategy_id, tool_skill_id, intent_bucket)`` — all strings.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..models import Candidate
from ..orchestrator import ContextRunState
from .models import SCHEMA_VERSION, OutcomeEvent, SignalType
from .redaction import hash_text


def workspace_fingerprint(workspace_root: Path | None) -> str:
    if workspace_root is None:
        return "global"
    try:
        return hash_text(str(workspace_root.resolve()))
    except OSError:
        return hash_text(str(workspace_root))


def cohort_key_from_state(state: ContextRunState) -> tuple[str, ...]:
    tier = state.execution_route.tier if state.execution_route else "unknown"
    cs = state.context_strategy or {}
    strategy_id = str(cs.get("strategy_id") or "")
    skill_id = ""
    if state.execution_route and state.execution_route.skill_id:
        skill_id = str(state.execution_route.skill_id)
    intent_bucket = str(state.intent)
    return (tier, strategy_id, skill_id, intent_bucket)


def _weak_scored_pool(pool: list[Candidate]) -> bool:
    """Mirror orchestrator weak-pool heuristic (retrieval quality proxy)."""
    if not pool:
        return True
    if not isinstance(pool[0], Candidate):
        return False
    top = float(pool[0].final_score)
    if len(pool) < 3 and top < 0.22:
        return True
    return top < 0.13


def _event_id_parts(
    *,
    ts: float,
    cohort: tuple[str, ...],
    signal: SignalType,
    wf: str,
    extra: str,
) -> str:
    payload = json.dumps(
        {"ts": ts, "cohort": cohort, "signal": signal.value, "wf": wf, "x": extra},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def build_events_from_run(state: ContextRunState) -> list[OutcomeEvent]:
    """Emit one ``local`` outcome event summarizing retrieval quality for this run."""
    cohort = cohort_key_from_state(state)
    wf = workspace_fingerprint(state.workspace_root)
    ts = time.time()
    pool = state.scored_pool
    weak = _weak_scored_pool(pool)
    top = float(pool[0].final_score) if pool else 0.0
    retry = bool(state.retrieval_retry_done)

    corr = {
        "session_id": state.session_id,
        "strategy_id": (state.context_strategy or {}).get("strategy_id"),
        "retry": retry,
    }

    events: list[OutcomeEvent] = []

    if weak:
        mag = min(1.0, max(0.25, 0.5 - top))
        if retry:
            mag = min(1.0, mag + 0.15)
        events.append(
            OutcomeEvent(
                schema_version=SCHEMA_VERSION,
                event_id=_event_id_parts(
                    ts=ts,
                    cohort=cohort,
                    signal=SignalType.RETRIEVAL_MISS_WEAK_POOL,
                    wf=wf,
                    extra="w",
                ),
                ts_epoch=ts,
                workspace_fingerprint=wf,
                source="local",
                signal_type=SignalType.RETRIEVAL_MISS_WEAK_POOL,
                magnitude=mag,
                cohort_key=cohort,
                correlation=corr,
            )
        )
    elif retry:
        events.append(
            OutcomeEvent(
                schema_version=SCHEMA_VERSION,
                event_id=_event_id_parts(
                    ts=ts,
                    cohort=cohort,
                    signal=SignalType.FOLLOW_UP_TIGHTENING,
                    wf=wf,
                    extra="r",
                ),
                ts_epoch=ts,
                workspace_fingerprint=wf,
                source="local",
                signal_type=SignalType.FOLLOW_UP_TIGHTENING,
                magnitude=0.35,
                cohort_key=cohort,
                correlation=corr,
            )
        )
    elif pool and top >= 0.22:
        events.append(
            OutcomeEvent(
                schema_version=SCHEMA_VERSION,
                event_id=_event_id_parts(
                    ts=ts,
                    cohort=cohort,
                    signal=SignalType.RETRIEVAL_HIT_STRONG,
                    wf=wf,
                    extra="h",
                ),
                ts_epoch=ts,
                workspace_fingerprint=wf,
                source="local",
                signal_type=SignalType.RETRIEVAL_HIT_STRONG,
                magnitude=min(1.0, top),
                cohort_key=cohort,
                correlation=corr,
            )
        )
    else:
        events.append(
            OutcomeEvent(
                schema_version=SCHEMA_VERSION,
                event_id=_event_id_parts(
                    ts=ts,
                    cohort=cohort,
                    signal=SignalType.BASELINE_TICK,
                    wf=wf,
                    extra="b",
                ),
                ts_epoch=ts,
                workspace_fingerprint=wf,
                source="local",
                signal_type=SignalType.BASELINE_TICK,
                magnitude=0.08,
                cohort_key=cohort,
                correlation=corr,
            )
        )

    return events
