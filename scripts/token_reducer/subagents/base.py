"""Lightweight pre-LLM subagents: filter, merge, rank — no extra model calls."""

from __future__ import annotations

from typing import Any

from ..models import Candidate


class SubAgent:
    """Deterministic chunk transform; gated by :meth:`should_run`."""

    name = "base"

    def should_run(self, intent: dict[str, Any], state: dict[str, Any]) -> bool:
        return False

    def run(self, chunks: list[Candidate], prompt: str, state: dict[str, Any]) -> list[Candidate]:
        return chunks
