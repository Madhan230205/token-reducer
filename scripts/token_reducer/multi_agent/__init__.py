"""True multi-agent orchestration: four specialists with non-overlapping pipeline roles.

See :mod:`token_reducer.multi_agent.agents` for Retriever, Reasoning enhancer, Compressor,

and Cost optimizer. Chunk-level deterministic passes remain in :mod:`token_reducer.subagents`.
"""

from __future__ import annotations

from .agents import (
    CompressorAgent,
    CostOptimizerAgent,
    ReasoningEnhancerAgent,
    RetrieverAgent,
)
from .runner import ORDERED_AGENTS, run_multi_agent_compression_phases

__all__ = [
    "CompressorAgent",
    "CostOptimizerAgent",
    "ORDERED_AGENTS",
    "ReasoningEnhancerAgent",
    "RetrieverAgent",
    "run_multi_agent_compression_phases",
]
