"""Bounded strategy nudges (§5.4): merge_cap ±20%, prune_k ±10% vs baseline."""

from __future__ import annotations

from .constants import MERGE_CAP_NUDGE_MAX_FRAC, PRUNE_K_NUDGE_MAX_FRAC


def clamp_merge_cap(baseline: int, proposed: int) -> int:
    lo = max(6, int(round(baseline * (1.0 - MERGE_CAP_NUDGE_MAX_FRAC))))
    hi = max(lo, int(round(baseline * (1.0 + MERGE_CAP_NUDGE_MAX_FRAC))))
    return max(lo, min(hi, proposed))


def clamp_prune_k(baseline: int, proposed: int) -> int:
    lo = max(4, int(round(baseline * (1.0 - PRUNE_K_NUDGE_MAX_FRAC))))
    hi = max(lo, int(round(baseline * (1.0 + PRUNE_K_NUDGE_MAX_FRAC))))
    return max(lo, min(hi, proposed))


def apply_task_shape_nudge(
    baseline_merge_cap: int,
    baseline_prune_k: int,
    task_shape: str,
) -> tuple[int, int]:
    """Apply heuristic multipliers from task_shape, then clamp to §5.4 bounds."""
    bm, bp = int(baseline_merge_cap), int(baseline_prune_k)
    mm, pm = 1.0, 1.0
    ts = task_shape.lower()
    if ts == "comparison":
        mm, pm = 1.10, 1.05
    elif ts == "generation":
        mm, pm = 1.15, 1.08
    elif ts == "iterative_refinement":
        mm, pm = 1.05, 1.10
    elif ts == "analysis":
        mm, pm = 1.05, 1.03
    elif ts == "single_shot":
        mm, pm = 0.90, 0.92
    prop_m = int(round(bm * mm))
    prop_p = int(round(bp * pm))
    return clamp_merge_cap(bm, prop_m), clamp_prune_k(bp, prop_p)
