"""Adaptive workspace feedback loop — bounded learning from outcomes (see design spec).

Design: docs/superpowers/specs/2026-05-02-adaptive-feedback-loop-design.md
Implementation plan: docs/superpowers/plans/2026-05-02-adaptive-feedback-loop.md

Environment (optional overrides):

- ``TOKEN_REDUCER_ADAPT_DISABLE`` — set to ``1``/``true`` to skip learning I/O and merged knobs.
- ``TOKEN_REDUCER_ADAPT_FLUSH_INTERVAL_MINUTES`` — debounce time gate (default 10).
- ``TOKEN_REDUCER_ADAPT_FLUSH_EVENT_BATCH`` — debounce mass gate (default 25).
- ``TOKEN_REDUCER_ADAPT_MIN_SAMPLES`` — minimum cohort samples before promotion (default 8).
- ``TOKEN_REDUCER_ADAPT_HOOK_WEIGHT`` — hook source weight, clamped to max 2.0 (default 1.75).

State files live under :func:`adaptive_dir` (workspace ``.token-reducer/adaptive/`` or global cache).
"""

from __future__ import annotations

from .attribution import ACTUATOR_FAMILIES, AttributionRow, attribution_for
from .constants import adaptive_disabled
from .debouncer import (
    DebouncerMeta,
    maybe_flush_after_event,
    maybe_schedule_flush,
    persist_staging_after_events,
    record_event_and_should_flush,
    record_events_and_maybe_flush,
    run_flush_pipeline,
)
from .event_log import (
    append_event,
    default_adapt_dir,
    default_events_path,
    events_path,
    iter_recent_events,
)
from .guardrails import clamp_committed
from .models import SCHEMA_VERSION, CommittedActuators, OutcomeEvent, SignalType, StagingState
from .redaction import bound_diagnostic, hash_text
from .scorer import apply_event
from .state_store import (
    adaptive_dir,
    load_committed,
    load_staging,
    promote_committed_atomic,
    save_staging,
)

__all__ = [
    "ACTUATOR_FAMILIES",
    "AttributionRow",
    "DebouncerMeta",
    "SCHEMA_VERSION",
    "SignalType",
    "OutcomeEvent",
    "StagingState",
    "CommittedActuators",
    "adaptive_disabled",
    "adaptive_dir",
    "append_event",
    "apply_event",
    "attribution_for",
    "bound_diagnostic",
    "clamp_committed",
    "default_adapt_dir",
    "default_events_path",
    "events_path",
    "hash_text",
    "iter_recent_events",
    "load_committed",
    "load_staging",
    "maybe_flush_after_event",
    "maybe_schedule_flush",
    "persist_staging_after_events",
    "promote_committed_atomic",
    "record_event_and_should_flush",
    "record_events_and_maybe_flush",
    "run_flush_pipeline",
    "save_staging",
]
