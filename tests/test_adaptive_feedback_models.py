"""Tests for adaptive_feedback models and constants."""

from __future__ import annotations

import pytest
from token_reducer.adaptive_feedback.constants import (
    adaptive_disabled,
    flush_event_batch,
    flush_interval_minutes,
    hook_source_weight,
    min_samples_actuation,
)
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


def test_constants_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOKEN_REDUCER_ADAPT_FLUSH_INTERVAL_MINUTES", raising=False)
    monkeypatch.delenv("TOKEN_REDUCER_ADAPT_FLUSH_EVENT_BATCH", raising=False)
    monkeypatch.delenv("TOKEN_REDUCER_ADAPT_MIN_SAMPLES", raising=False)
    monkeypatch.delenv("TOKEN_REDUCER_ADAPT_HOOK_WEIGHT", raising=False)
    monkeypatch.delenv("TOKEN_REDUCER_ADAPT_DISABLE", raising=False)
    assert flush_interval_minutes() == 10
    assert flush_event_batch() == 25
    assert min_samples_actuation() == 8
    assert hook_source_weight() == 1.75
    assert adaptive_disabled() is False


def test_adaptive_disabled_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKEN_REDUCER_ADAPT_DISABLE", "1")
    assert adaptive_disabled() is True
