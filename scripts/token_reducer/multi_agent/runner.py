"""Execute the multi-agent compression sequence in canonical order."""

from __future__ import annotations

from typing import Any

from .agents import (
    CompressorAgent,
    CostOptimizerAgent,
    ReasoningEnhancerAgent,
    RetrieverAgent,
)

ORDERED_AGENTS = (
    CostOptimizerAgent,
    RetrieverAgent,
    ReasoningEnhancerAgent,
    CompressorAgent,
)


def run_multi_agent_compression_phases(orch: Any) -> list[dict[str, Any]]:
    """Run prepare + retrieve + enhance + compress; return per-agent telemetry rows."""
    trace: list[dict[str, Any]] = []
    cost = CostOptimizerAgent()
    trace.append({"agent": cost.name, "phase": "prepare", **cost.run_prepare(orch)})
    ret = RetrieverAgent()
    trace.append({"agent": ret.name, **ret.run(orch)})
    enh = ReasoningEnhancerAgent()
    trace.append({"agent": enh.name, **enh.run(orch)})
    comp = CompressorAgent()
    trace.append({"agent": comp.name, **comp.run(orch)})
    return trace
