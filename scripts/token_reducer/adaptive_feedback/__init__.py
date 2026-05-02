"""Adaptive workspace feedback loop — bounded learning from outcomes (see design spec).

Design: docs/superpowers/specs/2026-05-02-adaptive-feedback-loop-design.md
Implementation plan: docs/superpowers/plans/2026-05-02-adaptive-feedback-loop.md
"""

from __future__ import annotations

from .attribution import ACTUATOR_FAMILIES, AttributionRow, attribution_for
from .constants import adaptive_disabled
from .models import SCHEMA_VERSION, CommittedActuators, OutcomeEvent, SignalType, StagingState

__all__ = [
    "ACTUATOR_FAMILIES",
    "AttributionRow",
    "SCHEMA_VERSION",
    "SignalType",
    "OutcomeEvent",
    "StagingState",
    "CommittedActuators",
    "adaptive_disabled",
    "attribution_for",
]
