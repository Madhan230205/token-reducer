"""Clamp committed actuator deltas before promotion (spec Section 8)."""

from __future__ import annotations

from copy import deepcopy

from .models import CommittedActuators

# Additive composition with legacy feedback_loop_adjustments — keep conservative.
_MAX_MULT_DELTA: float = 0.06
_MAX_RF_DELTA: float = 0.02
_MAX_PRUNE_BIAS: float = 0.35
_MAX_SKILL_PRIOR: float = 0.02


def clamp_committed(c: CommittedActuators) -> CommittedActuators:
    """Return a copy with deltas clamped to documented ceilings."""
    out = deepcopy(c)
    out.retrieval_scale_mult_delta = max(
        -_MAX_MULT_DELTA,
        min(_MAX_MULT_DELTA, out.retrieval_scale_mult_delta),
    )
    out.relevance_floor_delta = max(
        -_MAX_RF_DELTA,
        min(_MAX_RF_DELTA, out.relevance_floor_delta),
    )
    for sid, v in list(out.prune_bias_ema_delta.items()):
        out.prune_bias_ema_delta[sid] = max(-_MAX_PRUNE_BIAS, min(_MAX_PRUNE_BIAS, v))
    for sk, v in list(out.skill_prior_delta.items()):
        out.skill_prior_delta[sk] = max(-_MAX_SKILL_PRIOR, min(_MAX_SKILL_PRIOR, v))
    return out
