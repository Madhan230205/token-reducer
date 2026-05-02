"""Model-aware scaling for chunk budgets (shared by subagents + assertions)."""

from __future__ import annotations


def model_scale(model: str | None) -> float:
    m = (model or "sonnet").lower()
    if "haiku" in m:
        return 0.7
    if "opus" in m:
        return 1.08
    return 1.0


def effective_chunk_token_budget(token_budget: int, model: str | None) -> int:
    return max(200, int(int(token_budget) * model_scale(model)))
