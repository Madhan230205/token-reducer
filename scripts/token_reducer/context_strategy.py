"""Query → autonomous context strategy (no LLM): shapes retrieval depth and Claude-facing framing.

Exposed to the model as a short *attention_frame* only — no pipeline jargon, no “token reducer”.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ContextStrategy:
    """Per-run selection plan; drives early prune, fusion, neighborhood, and invisible UX."""

    strategy_id: str
    merge_cap: int
    prune_k: int
    skip_fusion: bool
    skip_neighborhood: bool
    attention_frame: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _wc(q: str) -> int:
    return len((q or "").split())


def map_query_to_strategy(
    query: str,
    structured: dict[str, Any],
    adaptive_tier: str,
    *,
    use_vector: bool,
) -> ContextStrategy:
    """Map query + intent + corpus tier → concrete knobs and one invisible editorial line for Claude."""
    coarse = str(structured.get("type", "analysis"))
    legacy = str(structured.get("legacy_intent", "explain_code"))
    q = (query or "").strip()
    words = _wc(q)

    merge_cap = 50
    prune_k = 15
    skip_fusion = False
    skip_neighborhood = False
    strategy_id = "balanced_workspace"

    if coarse == "chat" or words < 10:
        strategy_id = "conversational_light"
        prune_k = 6
        merge_cap = 24
        skip_fusion = True
        skip_neighborhood = True
        attention_frame = (
            "Answer from the user’s words first; treat any attached excerpts as optional background."
        )
    elif legacy == "bug_fix":
        strategy_id = "failure_adjacent"
        prune_k = 16 if use_vector else 12
        attention_frame = (
            "Prioritize excerpts closest to failures, errors, and guards; use other lines as supporting context."
        )
    elif legacy == "navigation":
        strategy_id = "path_and_symbol"
        prune_k = 10
        attention_frame = (
            "Anchor conclusions in the cited paths and symbols; prefer definitions over peripheral prose."
        )
    elif legacy == "feature_add":
        strategy_id = "surface_and_hooks"
        prune_k = 14
        attention_frame = (
            "Ground proposed changes in entry points, routes, and public surfaces shown in the excerpts."
        )
    elif legacy == "refactor":
        strategy_id = "structure_first"
        prune_k = 14
        attention_frame = (
            "Favor structural seams (modules, classes, repeated patterns) visible in the excerpts."
        )
    else:  # explain_code / analysis default
        strategy_id = "behavior_truth"
        prune_k = 14
        attention_frame = (
            "Treat the excerpts as the source of truth for behavior; call out uncertainty explicitly if gaps remain."
        )

    if adaptive_tier == "fts_only":
        merge_cap = min(merge_cap, 40)
        prune_k = max(6, prune_k - 2)
    elif adaptive_tier == "full_hybrid" and legacy == "bug_fix" and coarse != "chat":
        prune_k = min(18, prune_k + 1)

    return ContextStrategy(
        strategy_id=strategy_id,
        merge_cap=int(merge_cap),
        prune_k=int(prune_k),
        skip_fusion=bool(skip_fusion),
        skip_neighborhood=bool(skip_neighborhood),
        attention_frame=attention_frame,
    )
