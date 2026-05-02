"""Weighted EMA scoring into :class:`StagingState` (design spec Sections 5–6)."""

from __future__ import annotations

from typing import Final

from .constants import hook_source_weight, local_source_weight
from .models import OutcomeEvent, SignalType, StagingState

# Section 4 taxonomy — positive vs negative leaning for v1 utility/penalty split.
_POSITIVE: Final[frozenset[SignalType]] = frozenset(
    {
        SignalType.RETRIEVAL_HIT_STRONG,
        SignalType.COMPRESSION_ADEQUATE,
        SignalType.SESSION_FLOW_SMOOTH,
    }
)
_NEGATIVE: Final[frozenset[SignalType]] = frozenset(
    {
        SignalType.RETRIEVAL_MISS_WEAK_POOL,
        SignalType.FOLLOW_UP_TIGHTENING,
        SignalType.HOOK_TOOL_FAILURE,
    }
)

# Incremental EMA smoothing factor applied per event (tunable later via env).
_EMA_ALPHA: Final[float] = 0.12

# Gentle decay on baseline_tick for the event's cohort only.
_DECAY_FACTOR: Final[float] = 0.995


def _weight_for(event: OutcomeEvent) -> float:
    return hook_source_weight() if event.source == "hook" else local_source_weight()


def apply_event(state: StagingState, event: OutcomeEvent) -> None:
    """Apply one outcome: weighted EMA into staging; never raises."""
    cohort = event.cohort_key
    st = event.signal_type

    if st == SignalType.BASELINE_TICK:
        u = state.cohort_utility.get(cohort, 0.0)
        p = state.cohort_penalty.get(cohort, 0.0)
        state.cohort_utility[cohort] = u * _DECAY_FACTOR
        state.cohort_penalty[cohort] = p * _DECAY_FACTOR
        return

    x = max(0.0, min(1.0, event.magnitude)) * _weight_for(event)

    if st in _POSITIVE:
        old_u = state.cohort_utility.get(cohort, 0.0)
        state.cohort_utility[cohort] = _EMA_ALPHA * x + (1.0 - _EMA_ALPHA) * old_u
    elif st in _NEGATIVE:
        old_p = state.cohort_penalty.get(cohort, 0.0)
        state.cohort_penalty[cohort] = _EMA_ALPHA * x + (1.0 - _EMA_ALPHA) * old_p

    state.samples_per_cohort[cohort] = state.samples_per_cohort.get(cohort, 0) + 1
