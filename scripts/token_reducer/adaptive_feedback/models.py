"""Dataclasses and enums for the adaptive workspace feedback loop (spec: adapt_feedback_v1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

SCHEMA_VERSION = "adapt_feedback_v1"


class SignalType(StrEnum):
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
    """EMA nets per cohort — extended in scorer/actuators tasks."""

    cohort_utility: dict[tuple[str, ...], float] = field(default_factory=dict)
    cohort_penalty: dict[tuple[str, ...], float] = field(default_factory=dict)
    samples_per_cohort: dict[tuple[str, ...], int] = field(default_factory=dict)


@dataclass
class CommittedActuators:
    """Knobs the pipeline may read after promotion — v1 numeric biases only."""

    retrieval_scale_mult_delta: float = 0.0
    relevance_floor_delta: float = 0.0
    prune_bias_ema_delta: dict[str, float] = field(default_factory=dict)
    skill_prior_delta: dict[str, float] = field(default_factory=dict)
