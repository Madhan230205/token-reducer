"""Pluggable subagent list (single-pass, deterministic)."""

from __future__ import annotations

from .base import SubAgent

SUBAGENTS: list[SubAgent] = []


def register(agent: SubAgent) -> None:
    SUBAGENTS.append(agent)


def get_agents() -> list[SubAgent]:
    return list(SUBAGENTS)
