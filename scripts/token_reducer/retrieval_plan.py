"""Task mode + retrieval plan + query shaping (plan before chunk flood)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .intent import IntentType
from .repo_map import RepoMap

TaskMode = Literal["navigate", "explain", "debug", "refactor", "add_feature", "write_test"]

_WRITE_TEST = re.compile(
    r"\b(pytest|unittest|jest|mocha|write\s+test|add\s+test|unit\s+test|coverage)\b",
    re.I,
)
_DEPENDENCY_TRACE = re.compile(
    r"\b(stack\s+trace|traceback|caller|call\s+stack|depends?\s+on|referenced\s+from|"
    r"used\s+by|who\s+calls|call\s+site|invoked\s+from|import\s+chain)\b",
    re.I,
)
_DOWNSTREAM = re.compile(
    r"\b(propagate|downstream|consumers?|ripple|all\s+usages|every\s+caller|callee|"
    r"break\s+clients?)\b",
    re.I,
)
_CODEISH = re.compile(
    r"(\.(py|ts|tsx|js|jsx|go|rs|java)\b|\b(def|class|fn|func|import|from\s+\w|export\s+|interface\s+|struct\s+|impl\s+))",
    re.I,
)


@dataclass(frozen=True)
class RetrievalPlan:
    """What to fetch before / alongside chunk retrieval — keeps work plan-shaped."""

    effective_fts_query: str
    fts_cap: int
    effective_top_k: int
    source_priority: frozenset[str]
    second_pass_on_weak_pool: bool
    rewrite_on_second_pass: bool
    expand_neighbor_chunks: bool


def intent_to_task_mode(intent: IntentType, query: str) -> TaskMode:
    """Map legacy intent + query cues to a task mode for planning."""
    q = (query or "").strip()
    if _WRITE_TEST.search(q):
        return "write_test"
    if intent == "navigation":
        return "navigate"
    if intent == "explain_code":
        return "explain"
    if intent == "bug_fix":
        return "debug"
    if intent == "refactor":
        return "refactor"
    if intent == "feature_add":
        return "add_feature"
    return "explain"


def query_suggests_dependency_tracing(query: str) -> bool:
    """True when the user is asking about call paths, imports, or failure sites."""
    return bool(_DEPENDENCY_TRACE.search(query or ""))


def query_suggests_downstream_impact(query: str) -> bool:
    """True when edits may need to propagate to dependents."""
    return bool(_DOWNSTREAM.search(query or ""))


def query_suggests_code_exploration(query: str) -> bool:
    """True when explain-mode queries still benefit from LSP (paths, defs, modules)."""
    q = (query or "").strip()
    if len(q) < 8:
        return False
    if _CODEISH.search(q):
        return True
    ids = re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", q)
    return len(ids) >= 2


def plan_effective_fts_query(query: str, task_mode: str, *, rewrite: bool) -> str:
    """Cheap query shaping for FTS tokenization (not an LLM rewrite)."""
    q = (query or "").strip()
    if not q:
        return q
    if rewrite:
        q = re.sub(
            r"\b(please|kindly|the|a|an|just|can you|could you|how do i|how to)\b",
            " ",
            q,
            flags=re.I,
        )
        q = re.sub(r"\s+", " ", q).strip()
    if task_mode == "debug":
        q = re.sub(r"\b(my|our|this)\s+code\b", " ", q, flags=re.I)
        q = re.sub(r"\s+", " ", q).strip()
    return q


def task_source_flags(task_mode: str, query: str = "") -> dict[str, bool]:
    """Which repo roles matter for retrieval boosts — single source of truth."""
    q = query or ""
    include_callers = task_mode in ("debug", "refactor", "add_feature")
    if task_mode == "write_test":
        include_callers = query_suggests_dependency_tracing(q)
    include_callees = task_mode in ("debug", "add_feature", "write_test")
    if task_mode == "refactor":
        include_callees = query_suggests_downstream_impact(q)
    return {
        "include_tests": task_mode in ("debug", "write_test", "refactor"),
        "include_entry_points": task_mode in ("add_feature", "explain", "navigate", "debug"),
        "include_callers": include_callers,
        "include_callees": include_callees,
        "patch_first": task_mode in ("debug", "refactor", "add_feature", "write_test"),
    }


def build_retrieval_plan(
    task_mode: str,
    query: str,
    repo_map: RepoMap,
    fts_k: int,
    top_k: int,
    retrieval_depth: int,
    rewrite_query: bool,
    must_keep_tokens: frozenset[str],
) -> RetrievalPlan:
    """task_mode + repo_map -> concrete retrieval knobs (O(files) priority)."""
    flags = task_source_flags(task_mode, query)
    q1 = plan_effective_fts_query(query, task_mode, rewrite=False)
    pri = repo_map.boosted_sources_for_task(
        task_mode,
        must_keep_tokens,
        include_tests=flags["include_tests"],
        include_entry_points=flags["include_entry_points"],
        include_callers=flags["include_callers"],
        include_callees=flags["include_callees"],
        cap=72,
    )
    fts_cap = fts_k
    if task_mode in ("debug", "write_test"):
        fts_cap = min(int(fts_k * 1.25) + 6, 120)
    elif task_mode == "navigate":
        fts_cap = min(fts_k + 4, 96)

    eff_top = top_k
    if flags["patch_first"]:
        eff_top = max(3, min(top_k, top_k - 1))
    if task_mode == "navigate":
        eff_top = min(eff_top, 8)

    second = retrieval_depth >= 2
    expand_nb = flags["include_callers"] or flags["include_callees"]
    return RetrievalPlan(
        effective_fts_query=q1,
        fts_cap=fts_cap,
        effective_top_k=eff_top,
        source_priority=pri,
        second_pass_on_weak_pool=second,
        rewrite_on_second_pass=rewrite_query,
        expand_neighbor_chunks=expand_nb,
    )


def plan_boosted_sources(
    repo: RepoMap, task_mode: str, *, query: str = "", max_boost: int = 24
) -> frozenset[str]:
    """Backward-compatible boost set — prefer boosted_sources_for_task on RepoMap."""
    f = task_source_flags(task_mode, query)
    return repo.boosted_sources_for_task(
        task_mode,
        frozenset(),
        include_tests=f["include_tests"],
        include_entry_points=f["include_entry_points"],
        include_callers=f["include_callers"],
        include_callees=f["include_callees"],
        cap=max_boost,
    )


def query_must_keep_tokens(query: str, *, cap: int = 16) -> frozenset[str]:
    """Identifiers from the query to prefer preserving in code compression."""
    raw = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{1,48}\b", query or "")
    noise = frozenset(
        {
            "the",
            "and",
            "for",
            "with",
            "this",
            "that",
            "from",
            "when",
            "what",
            "where",
            "how",
            "why",
            "fix",
            "add",
            "use",
            "get",
            "set",
        }
    )
    out: list[str] = []
    for t in raw:
        tl = t.lower()
        if tl in noise or len(t) < 3:
            continue
        if t not in out:
            out.append(t)
        if len(out) >= cap:
            break
    return frozenset(out)
