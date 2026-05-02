"""Tests for debouncer."""

from __future__ import annotations

from pathlib import Path

import pytest
from token_reducer.adaptive_feedback.debouncer import (
    DebouncerMeta,
    load_meta,
    maybe_flush_after_event,
    record_event_and_should_flush,
    record_events_and_maybe_flush,
    save_meta,
)
from token_reducer.adaptive_feedback.models import StagingState
from token_reducer.adaptive_feedback.state_store import load_committed, save_staging


def test_record_event_mass_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOKEN_REDUCER_ADAPT_FLUSH_EVENT_BATCH", "2")
    monkeypatch.delenv("TOKEN_REDUCER_ADAPT_DISABLE", raising=False)
    root = tmp_path / "ws"
    save_staging(root, StagingState())

    assert maybe_flush_after_event(root, 100.0) is False
    assert maybe_flush_after_event(root, 101.0) is True
    meta = load_meta(root)
    assert meta.events_since_flush == 0
    committed = load_committed(root)
    assert committed is not None


def test_adaptive_disabled_skips_flush(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOKEN_REDUCER_ADAPT_DISABLE", "1")
    monkeypatch.setenv("TOKEN_REDUCER_ADAPT_FLUSH_EVENT_BATCH", "1")
    root = tmp_path / "ws"
    save_staging(root, StagingState())
    assert maybe_flush_after_event(root, 1.0) is False


def test_record_events_batch_increment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOKEN_REDUCER_ADAPT_FLUSH_EVENT_BATCH", "5")
    monkeypatch.delenv("TOKEN_REDUCER_ADAPT_DISABLE", raising=False)
    root = tmp_path / "ws"
    save_staging(root, StagingState())
    assert record_events_and_maybe_flush(root, 1.0, event_count=3) is False
    assert record_events_and_maybe_flush(root, 2.0, event_count=2) is True
    meta = load_meta(root)
    assert meta.events_since_flush == 0


def test_record_event_returns_meta_without_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKEN_REDUCER_ADAPT_FLUSH_EVENT_BATCH", "99")
    monkeypatch.delenv("TOKEN_REDUCER_ADAPT_DISABLE", raising=False)
    root = tmp_path / "ws2"
    save_meta(root, DebouncerMeta(last_flush_ts=0.0, events_since_flush=0))
    should, meta = record_event_and_should_flush(root, 0.0, interval_minutes=999, event_batch=99)
    assert should is False
    assert meta.events_since_flush == 1
