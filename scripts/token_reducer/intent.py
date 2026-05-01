"""Query intent classification for Claude-optimized retrieval."""

from __future__ import annotations

import re
from typing import Literal

IntentType = Literal["bug_fix", "feature_add", "explain_code", "refactor", "navigation"]

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
