"""Query intent classification for Claude-optimized retrieval."""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

IntentType = Literal["bug_fix", "feature_add", "explain_code", "refactor", "navigation"]

StructuredIntentType = Literal["code", "chat", "analysis"]
CompressionLevel = Literal["high", "medium", "low"]


class StructuredIntent(TypedDict):
    """Structured intent driving retrieval depth, token budget, and compression."""

    type: StructuredIntentType
    k: int
    token_budget: int
    compression_level: CompressionLevel
    legacy_intent: IntentType

_BUG = re.compile(
    r"\b(bug|crash|error|exception|traceback|stack\s*trace|fail|broken|fix|regression|"
    r"panic|undefined|null\s*reference|segfault|leak|timeout|500|404)\b",
    re.I,
)
_FEATURE = re.compile(
    r"\b(add|implement|new\s+feature|support|extend|create|build|introduce|"
    r"endpoint|api|route|handler|plugin)\b",
    re.I,
)
_EXPLAIN = re.compile(
    r"\b(what|how|why|explain|understand|describe|walk\s*through|meaning|"
    r"does\s+this|where\s+does)\b",
    re.I,
)
_REFACTOR = re.compile(
    r"\b(refactor|rename|extract|split|clean\s*up|dedupe|simplify|migrate|"
    r"restructure|modernize)\b",
    re.I,
)
_NAV = re.compile(
    r"\b(where|find|locate|goto|jump|open|which\s+file|path\s+to|navigate)\b",
    re.I,
)


def analyze_query_intent(query: str) -> IntentType:
    """Classify user query into intent using keyword heuristics."""
    q = (query or "").strip()
    if not q:
        return "explain_code"
    scores: dict[IntentType, int] = {
        "bug_fix": len(_BUG.findall(q)),
        "feature_add": len(_FEATURE.findall(q)),
        "explain_code": len(_EXPLAIN.findall(q)),
        "refactor": len(_REFACTOR.findall(q)),
        "navigation": len(_NAV.findall(q)),
    }
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        if re.search(r"\.(py|ts|js|go|rs|java)\b", q, re.I):
            return "navigation"
        return "explain_code"
    return best


def _structured_for_legacy(legacy: IntentType) -> tuple[StructuredIntentType, int, int, CompressionLevel]:
    """Map legacy intent to (coarse type, retrieval k, token_budget, compression_level)."""
    if legacy in ("bug_fix", "feature_add", "refactor", "navigation"):
        profiles: dict[IntentType, tuple[int, int, CompressionLevel]] = {
            "bug_fix": (52, 2400, "high"),
            "feature_add": (50, 2200, "high"),
            "refactor": (48, 2600, "medium"),
            "navigation": (36, 1800, "medium"),
        }
        k, budget, level = profiles[legacy]
        return "code", k, budget, level
    if legacy == "explain_code":
        return "analysis", 44, 2800, "medium"
    return "analysis", 40, 2000, "medium"


def detect_intent(prompt: str) -> StructuredIntent:
    """Structured intent for the dominant pipeline (retrieval k, token budget, compression).

    Always includes ``legacy_intent`` so existing routing (task_mode, cache keys) stays aligned
    with :func:`analyze_query_intent`.
    """
    q = (prompt or "").strip()
    legacy = analyze_query_intent(q)
    coarse, k, token_budget, compression_level = _structured_for_legacy(legacy)

    ql = q.lower()
    if any(x in ql for x in ("thanks", "hello", "hi ", "please write", "tone", "summarize chat")):
        coarse = "chat"
        k = min(k, 28)
        token_budget = min(token_budget, 1600)
        compression_level = "low"

    return StructuredIntent(
        type=coarse,
        k=int(k),
        token_budget=int(token_budget),
        compression_level=compression_level,
        legacy_intent=legacy,
    )


def structured_intent_to_dict(si: StructuredIntent) -> dict[str, Any]:
    """Plain dict copy for JSON/debug surfaces."""
    return {
        "type": si["type"],
        "k": si["k"],
        "token_budget": si["token_budget"],
        "compression_level": si["compression_level"],
        "legacy_intent": si["legacy_intent"],
    }
