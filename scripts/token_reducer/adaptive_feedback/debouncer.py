"""Debounced flush scheduling (spec Section 7: time T and mass M gates)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .actuators import committed_from_staging
from .constants import (
    adaptive_disabled,
    flush_event_batch,
    flush_interval_minutes,
    min_samples_actuation,
)
from .guardrails import clamp_committed
from .models import StagingState
from .state_store import adaptive_dir, load_staging, promote_committed_atomic, save_staging


@dataclass
class DebouncerMeta:
    last_flush_ts: float
    events_since_flush: int


def meta_path(workspace_root: Path | None) -> Path:
    return adaptive_dir(workspace_root) / "debouncer_meta.json"


def load_meta(workspace_root: Path | None) -> DebouncerMeta:
    path = meta_path(workspace_root)
    if not path.is_file():
        return DebouncerMeta(last_flush_ts=0.0, events_since_flush=0)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DebouncerMeta(last_flush_ts=0.0, events_since_flush=0)
    if not isinstance(raw, dict):
        return DebouncerMeta(last_flush_ts=0.0, events_since_flush=0)
    try:
        return DebouncerMeta(
            last_flush_ts=float(raw.get("last_flush_ts", 0.0)),
            events_since_flush=int(raw.get("events_since_flush", 0)),
        )
    except (TypeError, ValueError):
        return DebouncerMeta(last_flush_ts=0.0, events_since_flush=0)


def save_meta(workspace_root: Path | None, meta: DebouncerMeta) -> None:
    path = meta_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"last_flush_ts": meta.last_flush_ts, "events_since_flush": meta.events_since_flush},
        indent=2,
    )
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def record_event_and_should_flush(
    workspace_root: Path | None,
    now_ts: float,
    *,
    interval_minutes: int | None = None,
    event_batch: int | None = None,
) -> tuple[bool, DebouncerMeta]:
    """Increment event counter; return ``(should_flush, updated_meta)``."""
    meta = load_meta(workspace_root)
    meta.events_since_flush += 1
    interval = flush_interval_minutes() if interval_minutes is None else interval_minutes
    batch = flush_event_batch() if event_batch is None else event_batch

    if meta.last_flush_ts <= 0.0:
        should = meta.events_since_flush >= batch
    else:
        elapsed_min = (now_ts - meta.last_flush_ts) / 60.0
        should = meta.events_since_flush >= batch or elapsed_min >= float(interval)

    return should, meta


def reset_meta_after_flush(workspace_root: Path | None, now_ts: float) -> None:
    save_meta(workspace_root, DebouncerMeta(last_flush_ts=now_ts, events_since_flush=0))


def run_flush_pipeline(workspace_root: Path | None, now_ts: float) -> bool:
    """Promote staging → committed when gates fire. Returns True if a flush ran."""
    if adaptive_disabled():
        return False
    staging = load_staging(workspace_root)
    committed = clamp_committed(
        committed_from_staging(staging, min_samples=min_samples_actuation())
    )
    promote_committed_atomic(workspace_root, committed)
    reset_meta_after_flush(workspace_root, now_ts)
    return True


def maybe_flush_after_event(workspace_root: Path | None, now_ts: float) -> bool:
    """Increment debouncer counter; flush (promote + reset meta) when mass/time gates hit."""
    if adaptive_disabled():
        return False
    should, meta = record_event_and_should_flush(workspace_root, now_ts)
    if should:
        return run_flush_pipeline(workspace_root, now_ts)
    save_meta(workspace_root, meta)
    return False


def maybe_schedule_flush(workspace_root: Path | None, now_ts: float) -> bool:
    """Alias for pipeline hooks (plan naming)."""
    return maybe_flush_after_event(workspace_root, now_ts)


def persist_staging_after_events(workspace_root: Path | None, staging: StagingState) -> None:
    """Persist scorer staging (call after applying events in memory)."""
    if adaptive_disabled():
        return
    save_staging(workspace_root, staging)
