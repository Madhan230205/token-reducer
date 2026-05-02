"""Tests for adaptive_feedback event log and redaction."""

from __future__ import annotations

from pathlib import Path

from token_reducer.adaptive_feedback.event_log import append_event, iter_recent_events
from token_reducer.adaptive_feedback.models import SCHEMA_VERSION, OutcomeEvent, SignalType
from token_reducer.adaptive_feedback.redaction import bound_diagnostic, hash_text


def test_hash_text_stable() -> None:
    assert hash_text("hello") == hash_text("hello")
    assert hash_text("a") != hash_text("b")


def test_bound_diagnostic_truncates() -> None:
    long = "x" * 200
    assert len(bound_diagnostic(long, max_len=50)) <= 50


def test_iter_recent_events_skips_bad_lines(tmp_path: Path) -> None:
    log = tmp_path / "e.jsonl"
    log.write_text(
        "not json\n"
        + '{"schema_version": "wrong", "event_id": "1"}\n'
        + '{"schema_version": "adapt_feedback_v1", "event_id": "a", "ts_epoch": 1.0, '
        '"workspace_fingerprint": "fp", "source": "local", "signal_type": '
        '"baseline_tick", "magnitude": 0.1, "cohort_key": ["t","s","n","i"]}\n'
        + '{"schema_version": "adapt_feedback_v1", "event_id": "b", "ts_epoch": 2.0, '
        '"workspace_fingerprint": "fp", "source": "hook", "signal_type": '
        '"hook_tool_failure", "magnitude": 0.5, "cohort_key": ["t","s","n","i"]}\n',
        encoding="utf-8",
    )
    evs = iter_recent_events(log, tail_lines=50)
    assert len(evs) == 2
    assert evs[0].event_id == "a"
    assert evs[1].source == "hook"


def test_append_roundtrip(tmp_path: Path) -> None:
    log = tmp_path / "out.jsonl"
    ev = OutcomeEvent(
        schema_version=SCHEMA_VERSION,
        event_id="id1",
        ts_epoch=3.0,
        workspace_fingerprint="wf",
        source="local",
        signal_type=SignalType.RETRIEVAL_HIT_STRONG,
        magnitude=0.8,
        cohort_key=("tool", "st", "sk", "intent"),
        correlation={"x": 1},
    )
    append_event(log, ev)
    back = iter_recent_events(log)
    assert len(back) == 1
    assert back[0].cohort_key == ("tool", "st", "sk", "intent")
    assert back[0].correlation == {"x": 1}
