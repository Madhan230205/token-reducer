"""Adaptive workspace feedback loop — bounded learning from outcomes (see design spec).

Design: docs/superpowers/specs/2026-05-02-adaptive-feedback-loop-design.md
Implementation plan: docs/superpowers/plans/2026-05-02-adaptive-feedback-loop.md
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
    run_flush_pipeline,
)
from .event_log import append_event, default_adapt_dir, default_events_path, iter_recent_events
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
    "hash_text",
    "iter_recent_events",
    "load_committed",
    "load_staging",
    "maybe_flush_after_event",
    "maybe_schedule_flush",
    "persist_staging_after_events",
    "promote_committed_atomic",
    "record_event_and_should_flush",
    "run_flush_pipeline",
    "save_staging",
]
