"""Tests for adaptive_feedback scorer."""

from __future__ import annotations

import pytest
from token_reducer.adaptive_feedback.models import OutcomeEvent, SignalType, StagingState
from token_reducer.adaptive_feedback.scorer import apply_event


def _ev(source: str, cohort: tuple[str, ...] = ("a", "b", "c", "d")) -> OutcomeEvent:
    return OutcomeEvent(
        schema_version="adapt_feedback_v1",
        event_id="e1",
        ts_epoch=1.0,
        workspace_fingerprint="wf",
        source=source,  # type: ignore[arg-type]
        signal_type=SignalType.RETRIEVAL_HIT_STRONG,
        magnitude=1.0,
        cohort_key=cohort,
        correlation=None,
    )


def test_hook_weight_stronger_than_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKEN_REDUCER_ADAPT_HOOK_WEIGHT", "2.0")
    monkeypatch.delenv("TOKEN_REDUCER_ADAPT_DISABLE", raising=False)

    slocal = StagingState()
    apply_event(slocal, _ev("local"))
    shout = StagingState()
    apply_event(shout, _ev("hook"))

    assert shout.cohort_utility[("a", "b", "c", "d")] > slocal.cohort_utility[("a", "b", "c", "d")]


def test_baseline_tick_decay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOKEN_REDUCER_ADAPT_HOOK_WEIGHT", raising=False)
    st = StagingState()
    st.cohort_utility[("a",)] = 1.0
    st.cohort_penalty[("a",)] = 0.5
    ev = OutcomeEvent(
        schema_version="adapt_feedback_v1",
        event_id="e2",
        ts_epoch=1.0,
        workspace_fingerprint="wf",
        source="local",
        signal_type=SignalType.BASELINE_TICK,
        magnitude=0.0,
        cohort_key=("a",),
        correlation=None,
    )
    apply_event(st, ev)
    assert st.cohort_utility[("a",)] < 1.0
    assert st.cohort_penalty[("a",)] < 0.5
