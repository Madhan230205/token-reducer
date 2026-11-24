"""Validate ranking weight overrides (§3.5)."""

from __future__ import annotations

from typing import Any

from .constants import WEIGHT_MAX_DIM, WEIGHT_SUM_EPS


def validate_ranking_weights(raw: dict[str, Any] | None) -> dict[str, float] | None:
    """Return normalized weights or None if invalid."""
    if not raw:
        return None
    try:
        items = {str(k): float(v) for k, v in raw.items()}
    except (TypeError, ValueError):
        return None
    if not items:
        return None
    if any(v < 0 for v in items.values()):
        return None
    if any(v > WEIGHT_MAX_DIM + 1e-9 for v in items.values()):
        return None
    s = sum(items.values())
    if s <= 0:
        return None
    if abs(s - 1.0) > WEIGHT_SUM_EPS:
        return None
    return items
