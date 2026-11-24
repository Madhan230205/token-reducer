"""Layer-1 execution tier + Layer-2 routing hints (pattern/heuristic; optional env override).

Tiers: SIMPLE (no subagents), TOOL (single skill path), COMPLEX (full pipeline).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .feedback import median_recent_context_words
from .intent import IntentType, StructuredIntent

ExecutionTier = Literal["simple", "tool", "complex"]

_REDUCER_TOKEN_THRESHOLD_DEFAULT = 2800

_COMPLEX_MARKERS = re.compile(
    r"\b(step\s+by\s+step|multi[- ]step|whole\s+(codebase|repo|project)|refactor\s+(everything|the\s+entire)|"
    r"implement\s+(all|everything)|migration\s+plan|architecture\s+for)\b",
    re.I,
)
_TOOL_MARKERS = re.compile(
    r"^\s*/\w+|\b(invoke|run)\s+(the\s+)?skill\b|\b/use\s+skill\b",
    re.I,
)


@dataclass(frozen=True)
class ExecutionRoute:
    """Resolved route for one user turn (cheap to compute; drives orchestrator gates)."""

    tier: ExecutionTier
    skill_id: str | None
    skip_subagents: bool
    skip_lsp: bool
    retrieval_scale: float
    reducer_token_threshold: int
    efficiency_score: float | None  # registry metric when TOOL skill picked


def _plugin_root() -> Path | None:
    root = os.environ.get("CLAUDE_PLUGIN_ROOT", "").strip()
    return Path(root) if root else None


def load_skill_registry() -> dict[str, Any]:
    """Load skill_registry.json from plugin root (skills/)."""
    root = _plugin_root()
    candidates: list[Path] = []
    if root is not None:
        candidates.append(root / "skills" / "skill_registry.json")
    here = Path(__file__).resolve().parents[2]
    candidates.append(here / "skills" / "skill_registry.json")

    for path in candidates:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except (OSError, json.JSONDecodeError):
                break
    return {}


def _skill_efficiency(entry: dict[str, Any]) -> float:
    cost = float(entry.get("cost_estimate") or 1.0)
    lat = float(entry.get("latency_ms") or 100.0)
    succ = float(entry.get("success_rate") or 0.5)
    succ = max(0.05, min(1.0, succ))
    return (cost * lat) / succ


def select_skill_for_query(query: str, registry: dict[str, Any] | None = None) -> tuple[str | None, float | None]:
    """Pick one skill for TOOL tier by overlap with triggers × efficiency."""
    reg = registry if registry is not None else load_skill_registry()
    skills = reg.get("skills")
    if not isinstance(skills, list):
        return None, None
    q = (query or "").lower()
    best_id: str | None = None
    best_eff: float | None = None
    best_score = float("inf")
    for raw in skills:
        if not isinstance(raw, dict):
            continue
        sid = str(raw.get("id") or raw.get("name") or "").strip()
        if not sid:
            continue
        triggers = raw.get("triggers") or []
        overlap = 0
        if isinstance(triggers, list):
            for t in triggers:
                if isinstance(t, str) and t.lower() in q:
                    overlap += 1
        eff = _skill_efficiency(raw)
        score = eff / (1.0 + overlap)
        if overlap > 0 and score < best_score:
            best_score = score
            best_id = sid
            best_eff = eff
    if best_id is None:
        # Default: lowest-efficiency skill as neutral TOOL anchor
        for raw in skills:
            if not isinstance(raw, dict):
                continue
            sid = str(raw.get("id") or raw.get("name") or "").strip()
            if not sid:
                continue
            eff = _skill_efficiency(raw)
            if best_eff is None or eff < best_eff:
                best_eff = eff
                best_id = sid
    return best_id, best_eff


def tier_override_from_env() -> ExecutionTier | None:
    """Inject tier without LLM (CI, outer orchestrator)."""
    raw = (os.environ.get("TOKEN_REDUCER_TIER_OVERRIDE") or "").strip().lower()
    if raw in ("simple", "tool", "complex"):
        return raw  # type: ignore[return-value]
    return None


def classify_execution_tier(
    query: str,
    structured: StructuredIntent | dict[str, Any],
    *,
    legacy_intent: IntentType | None = None,
) -> ExecutionTier:
    """Layer-1 classifier: patterns + legacy intent; strict cheap path default."""
    forced = tier_override_from_env()
    if forced is not None:
        return forced

    q = (query or "").strip()
    words = q.split()
    n_words = len(words)
    n_lines = q.count("\n") + 1 if q else 0

    if isinstance(structured, dict):
        coarse = str(structured.get("type", ""))
        leg = structured.get("legacy_intent", legacy_intent)
    else:
        coarse = structured["type"]
        leg = structured["legacy_intent"]
    if legacy_intent is None and isinstance(leg, str):
        legacy_intent = leg  # type: ignore[assignment]

    if coarse == "chat" and n_words <= 48:
        return "simple"

    if _TOOL_MARKERS.search(q):
        return "tool"

    codeish = bool(re.search(r"[`][/\\]|\.(?:py|ts|js|tsx|jsx|go|rs|java)\b", q, re.I))
    if legacy_intent == "navigation" and n_words < 44 and not _COMPLEX_MARKERS.search(q):
        return "tool"

    if _COMPLEX_MARKERS.search(q):
        return "complex"
    if n_lines > 10 and n_words > 120:
        return "complex"
    if codeish and n_words > 55 and legacy_intent in ("bug_fix", "feature_add", "refactor"):
        return "complex"
    if legacy_intent in ("bug_fix", "feature_add", "refactor") and n_words > 42:
        return "complex"

    paths = len(re.findall(r"[/\\][\w.\-/\\]+\.(?:py|ts|js|go|rs|java)", q))
    if paths >= 2 and n_words > 30:
        return "complex"

    # Short, non-code prompts → cheapest path (80/20)
    if n_words <= 28 and not codeish and n_lines <= 5:
        return "simple"

    if n_words <= 40 and legacy_intent == "explain_code" and not codeish:
        return "simple"

    return "complex"


def resolve_execution_route(
    query: str,
    structured: StructuredIntent | dict[str, Any],
    *,
    legacy_intent: IntentType | None = None,
) -> ExecutionRoute:
    """Full Layer-2 route: tier, gates, optional single skill."""
    tier = classify_execution_tier(query, structured, legacy_intent=legacy_intent)
    thresh = int(os.environ.get("TOKEN_REDUCER_REDUCER_THRESHOLD") or _REDUCER_TOKEN_THRESHOLD_DEFAULT)

    if tier == "simple":
        return ExecutionRoute(
            tier=tier,
            skill_id=None,
            skip_subagents=True,
            skip_lsp=True,
            retrieval_scale=0.52,
            reducer_token_threshold=thresh,
            efficiency_score=None,
        )
    if tier == "tool":
        sid, eff = select_skill_for_query(query)
        return ExecutionRoute(
            tier=tier,
            skill_id=sid,
            skip_subagents=True,
            skip_lsp=False,
            retrieval_scale=0.72,
            reducer_token_threshold=thresh,
            efficiency_score=eff,
        )
    return ExecutionRoute(
        tier="complex",
        skill_id=None,
        skip_subagents=False,
        skip_lsp=False,
        retrieval_scale=1.0,
        reducer_token_threshold=thresh,
        efficiency_score=None,
    )


def execution_route_to_dict(route: ExecutionRoute) -> dict[str, Any]:
    return {
        "tier": route.tier,
        "skill_id": route.skill_id,
        "skip_subagents": route.skip_subagents,
        "skip_lsp": route.skip_lsp,
        "retrieval_scale": route.retrieval_scale,
        "reducer_token_threshold": route.reducer_token_threshold,
        "efficiency_score": route.efficiency_score,
    }


SubagentProfile = Literal["none", "lean", "full"]


@dataclass(frozen=True)
class RoutingPlan:
    """Hard output token cap + subagent chain hint for one turn."""

    output_token_cap: int
    subagent_profile: SubagentProfile
    tier: str


def _tier_budget_multiplier(tier: str) -> float:
    return {"simple": 0.78, "tool": 0.92, "complex": 1.05}.get(tier, 1.0)


def _parse_env_positive_int(name: str) -> int | None:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def _prefer_lean_subagents(workspace_root: Path | None) -> bool:
    _ = workspace_root
    if (os.environ.get("TOKEN_REDUCER_LEAN_SUBAGENTS") or "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    med = median_recent_context_words()
    return med is not None and med > 200


def build_routing_plan(
    route: ExecutionRoute,
    structured_intent: dict[str, Any],
    *,
    workspace_root: Path | None = None,
) -> RoutingPlan:
    base_tb = int(structured_intent.get("token_budget") or 2000)
    tier = str(route.tier)
    cap = max(320, int(round(base_tb * _tier_budget_multiplier(tier))))
    env_cap = _parse_env_positive_int("TOKEN_REDUCER_OUTPUT_TOKEN_CAP")
    if env_cap is not None:
        cap = min(cap, env_cap)

    if route.skip_subagents:
        profile: SubagentProfile = "none"
    elif tier == "complex":
        profile = "lean" if _prefer_lean_subagents(workspace_root) else "full"
    else:
        profile = "none"

    return RoutingPlan(output_token_cap=cap, subagent_profile=profile, tier=tier)


def routing_plan_to_dict(plan: RoutingPlan) -> dict[str, Any]:
    return {
        "output_token_cap": plan.output_token_cap,
        "subagent_profile": plan.subagent_profile,
        "tier": plan.tier,
    }
