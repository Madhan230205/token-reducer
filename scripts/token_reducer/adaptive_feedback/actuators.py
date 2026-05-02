"""Map staged cohort scores → committed actuator deltas (spec Section 6)."""

from __future__ import annotations

from .models import CommittedActuators, StagingState


def committed_from_staging(staging: StagingState, *, min_samples: int) -> CommittedActuators:
    """Aggregate per-cohort utility/penalty into global actuator deltas.

    Returns neutral :class:`CommittedActuators` if any cohort touched by staging has
    ``samples_per_cohort < min_samples`` (cold-start, spec Section 5.2).
    """
    touched = (
        set(staging.cohort_utility.keys())
        | set(staging.cohort_penalty.keys())
        | set(staging.samples_per_cohort.keys())
    )
    if not touched:
        return CommittedActuators()

    for k in touched:
        if staging.samples_per_cohort.get(k, 0) < min_samples:
            return CommittedActuators()

    nets: list[float] = []
    for k in touched:
        u = staging.cohort_utility.get(k, 0.0)
        p = staging.cohort_penalty.get(k, 0.0)
        nets.append(u - p)

    net = sum(nets) / len(nets)
    # net positive → lean outcomes → slight loosen; negative → tighten (matches feedback intuition).
    mult_delta = -0.035 * net
    rf_delta = 0.012 * net
    return CommittedActuators(
        retrieval_scale_mult_delta=mult_delta,
        relevance_floor_delta=rf_delta,
        prune_bias_ema_delta={},
        skill_prior_delta={},
    )
