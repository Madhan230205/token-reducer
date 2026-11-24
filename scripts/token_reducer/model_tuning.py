"""Upstream model hints for budgets + compression (no LLM calls; avoids subagents import cycles)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompressionModelTuning:
    """Knobs for packing context under a named upstream model (Sonnet baseline)."""

    budget_factor: float
    rank_sim_scale: float
    relevance_floor_shift: float
    prose_query_weight_scale: float


def model_scale(model: str | None) -> float:
    m = (model or "sonnet").lower()
    if "haiku" in m:
        return 0.7
    if "opus" in m:
        return 1.08
    return 1.0


def compression_model_tuning(model: str | None) -> CompressionModelTuning:
    """Haiku: tighter + query-first. Opus: slightly roomier + trust centrality more."""
    m = (model or "sonnet").lower()
    if "haiku" in m:
        return CompressionModelTuning(
            budget_factor=0.9,
            rank_sim_scale=1.16,
            relevance_floor_shift=0.032,
            prose_query_weight_scale=1.18,
        )
    if "opus" in m:
        return CompressionModelTuning(
            budget_factor=1.045,
            rank_sim_scale=0.93,
            relevance_floor_shift=-0.014,
            prose_query_weight_scale=0.91,
        )
    return CompressionModelTuning(
        budget_factor=1.0,
        rank_sim_scale=1.0,
        relevance_floor_shift=0.0,
        prose_query_weight_scale=1.0,
    )


def effective_chunk_token_budget(token_budget: int, model: str | None) -> int:
    return max(200, int(int(token_budget) * model_scale(model)))
