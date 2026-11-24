"""Protocols for the four top-level pipeline specialists (implementation-free)."""

from __future__ import annotations

from typing import Any, Protocol


class PipelineSpecialist(Protocol):
    """Runs one bounded phase against a :class:`~token_reducer.orchestrator.ContextPipelineOrchestrator`."""

    name: str

    def run(self, orch: Any) -> dict[str, Any]:
        ...
