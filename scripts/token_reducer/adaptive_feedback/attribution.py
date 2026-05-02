"""Normative signal_type → actuator family mapping (design spec Section 6.1)."""

from __future__ import annotations

from dataclasses import dataclass

from .models import SignalType

# Subset labels aligned with design doc Section 6 actuator list.
ACTUATOR_FAMILIES = frozenset({"feedback_multipliers", "strategy_prune", "skill_priors"})


@dataclass(frozen=True)
class AttributionRow:
    targets: frozenset[str]
    ambiguous: bool


def attribution_for(signal_type: SignalType) -> AttributionRow:
    """Return allowed actuator families for this signal; ambiguous → tighter caps in guardrails."""
    st = signal_type
    fm = frozenset({"feedback_multipliers"})
    sp = frozenset({"strategy_prune"})
    sk = frozenset({"skill_priors"})
    table: dict[SignalType, AttributionRow] = {
        SignalType.RETRIEVAL_HIT_STRONG: AttributionRow(fm, False),
        SignalType.RETRIEVAL_MISS_WEAK_POOL: AttributionRow(fm | sp, False),
        SignalType.COMPRESSION_ADEQUATE: AttributionRow(fm, False),
        SignalType.SESSION_FLOW_SMOOTH: AttributionRow(sk, False),
        SignalType.FOLLOW_UP_TIGHTENING: AttributionRow(fm | sp | sk, True),
        SignalType.HOOK_TOOL_FAILURE: AttributionRow(sk | fm, False),
        SignalType.BASELINE_TICK: AttributionRow(frozenset(), False),
    }
    return table[st]
