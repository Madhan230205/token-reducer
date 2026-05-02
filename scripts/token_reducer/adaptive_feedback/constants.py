"""Defaults and env-driven knobs for adaptive_feedback."""

from __future__ import annotations

import os

DEFAULT_FLUSH_INTERVAL_MINUTES = 10
DEFAULT_FLUSH_EVENT_BATCH = 25
DEFAULT_MIN_SAMPLES = 8
DEFAULT_HOOK_WEIGHT = 1.75
DEFAULT_HOOK_WEIGHT_MAX = 2.0
DEFAULT_LOCAL_WEIGHT = 1.0


def env_positive_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def flush_interval_minutes() -> int:
    return env_positive_int(
        "TOKEN_REDUCER_ADAPT_FLUSH_INTERVAL_MINUTES",
        DEFAULT_FLUSH_INTERVAL_MINUTES,
    )


def flush_event_batch() -> int:
    return env_positive_int("TOKEN_REDUCER_ADAPT_FLUSH_EVENT_BATCH", DEFAULT_FLUSH_EVENT_BATCH)


def min_samples_actuation() -> int:
    return env_positive_int("TOKEN_REDUCER_ADAPT_MIN_SAMPLES", DEFAULT_MIN_SAMPLES)


def hook_source_weight() -> float:
    raw = (os.environ.get("TOKEN_REDUCER_ADAPT_HOOK_WEIGHT") or "").strip()
    if not raw:
        w = DEFAULT_HOOK_WEIGHT
    else:
        try:
            w = float(raw)
        except ValueError:
            w = DEFAULT_HOOK_WEIGHT
    return max(1.0, min(DEFAULT_HOOK_WEIGHT_MAX, w))


def local_source_weight() -> float:
    return DEFAULT_LOCAL_WEIGHT


def adaptive_disabled() -> bool:
    return (os.environ.get("TOKEN_REDUCER_ADAPT_DISABLE") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
