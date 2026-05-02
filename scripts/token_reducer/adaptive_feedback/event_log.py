"""Append-only JSONL outcome log (design spec Section 3)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import SCHEMA_VERSION, OutcomeEvent, SignalType, SourceKind
from .state_store import adaptive_dir


def _as_float(obj: object) -> float | None:
    if isinstance(obj, bool):
        return None
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            return None
    return None


def default_adapt_dir() -> Path:
    """Same cache root family as feedback.jsonl; subdirectory ``adaptive``."""
    root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if root:
        return Path(root) / ".cache" / "token-reducer" / "adaptive"
    return Path.home() / ".cache" / "token-reducer" / "adaptive"


def default_events_path() -> Path:
    return default_adapt_dir() / "adapt_events.jsonl"


def events_path(workspace_root: Path | None) -> Path:
    """Per-workspace JSONL next to staging/committed when ``workspace_root`` is set."""
    return adaptive_dir(workspace_root) / "adapt_events.jsonl"


def event_to_row(event: OutcomeEvent) -> dict[str, object]:
    return {
        "schema_version": event.schema_version,
        "event_id": event.event_id,
        "ts_epoch": event.ts_epoch,
        "workspace_fingerprint": event.workspace_fingerprint,
        "source": event.source,
        "signal_type": event.signal_type.value,
        "magnitude": event.magnitude,
        "cohort_key": list(event.cohort_key),
        "correlation": event.correlation,
    }


def row_to_event(row: dict[str, object]) -> OutcomeEvent | None:
    try:
        if row.get("schema_version") != SCHEMA_VERSION:
            return None
        ck = row.get("cohort_key")
        if not isinstance(ck, list):
            return None
        cohort_tuple = tuple(str(x) for x in ck)
        sig = row.get("signal_type")
        if not isinstance(sig, str):
            return None
        try:
            signal_type = SignalType(sig)
        except ValueError:
            return None
        src = row.get("source")
        if src == "local":
            source: SourceKind = "local"
        elif src == "hook":
            source = "hook"
        else:
            return None
        corr_raw = row.get("correlation")
        correlation: dict[str, Any] | None
        if isinstance(corr_raw, dict):
            correlation = {str(k): v for k, v in corr_raw.items()}
        else:
            correlation = None
        ts = _as_float(row.get("ts_epoch"))
        mag = _as_float(row.get("magnitude"))
        if ts is None or mag is None:
            return None
        return OutcomeEvent(
            schema_version=str(row["schema_version"]),
            event_id=str(row["event_id"]),
            ts_epoch=ts,
            workspace_fingerprint=str(row["workspace_fingerprint"]),
            source=source,
            signal_type=signal_type,
            magnitude=mag,
            cohort_key=cohort_tuple,
            correlation=correlation,
        )
    except (KeyError, TypeError, ValueError):
        return None


def append_event(path: Path | None, event: OutcomeEvent) -> None:
    """Append one JSON line; swallow ``OSError`` (spec: never crash pipeline)."""
    target = path or default_events_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event_to_row(event), ensure_ascii=False)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        return


def iter_recent_events(path: Path | None, *, tail_lines: int = 500) -> list[OutcomeEvent]:
    """Parse last ``tail_lines`` lines; skip malformed JSON / invalid rows."""
    target = path or default_events_path()
    if not target.is_file():
        return []
    try:
        raw = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out: list[OutcomeEvent] = []
    for line in raw[-tail_lines:]:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        ev = row_to_event(obj)
        if ev is not None:
            out.append(ev)
    return out
