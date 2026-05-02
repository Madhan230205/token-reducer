"""Tests for signal → actuator attribution."""

from __future__ import annotations

from token_reducer.adaptive_feedback.attribution import ACTUATOR_FAMILIES, attribution_for
from token_reducer.adaptive_feedback.models import SignalType


def test_each_signal_has_row() -> None:
    for st in SignalType:
        row = attribution_for(st)
        assert row.targets <= ACTUATOR_FAMILIES
        assert isinstance(row.ambiguous, bool)


def test_baseline_tick_targets_empty() -> None:
    assert attribution_for(SignalType.BASELINE_TICK).targets == frozenset()


def test_follow_up_tightening_ambiguous() -> None:
    row = attribution_for(SignalType.FOLLOW_UP_TIGHTENING)
    assert row.ambiguous is True
    assert row.targets == ACTUATOR_FAMILIES


def test_retrieval_hit_strong_only_feedback_multipliers() -> None:
    row = attribution_for(SignalType.RETRIEVAL_HIT_STRONG)
    assert row.targets == frozenset({"feedback_multipliers"})
    assert row.ambiguous is False
