"""Load scenarios, run pipeline, emit benchmark_proof_v1 JSON Lines rows."""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from token_reducer.chunker import estimate_tokens
from token_reducer.db import connect_db, upsert_document
from token_reducer.execution_route import build_routing_plan, resolve_execution_route
from token_reducer.intent import analyze_query_intent, detect_intent, structured_intent_to_dict
from token_reducer.multi_agent.agents import (
    CompressorAgent,
    CostOptimizerAgent,
    ReasoningEnhancerAgent,
    RetrieverAgent,
)
from token_reducer.orchestrator import ContextPipelineOrchestrator, ContextRunState
from token_reducer.plugin_settings import get_runtime_defaults

from .stability import canonical_bullets_hash, canonical_selected_sources_hash, plugin_subset_digest

SCHEMA_VERSION = "benchmark_proof_v1"


@dataclass(frozen=True)
class BenchmarkScenario:
    scenario_id: str
    tier: str
    fixture_id: str
    query: str
    env: dict[str, Any]
    expected_paths: list[str]
    required: bool
    tags: list[str]
    stability_hash: dict[str, Any] | None


def _repo_git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GITHUB_SHA", "unknown")[:40] or "unknown"


def load_scenarios(scenarios_dir: Path) -> list[BenchmarkScenario]:
    out: list[BenchmarkScenario] = []
    for path in sorted(scenarios_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        tags = raw.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        env = raw.get("env") or {}
        if not isinstance(env, dict):
            env = {}
        stability = raw.get("stability_hash")
        if stability is not None and not isinstance(stability, dict):
            stability = None
        out.append(
            BenchmarkScenario(
                scenario_id=str(raw["scenario_id"]),
                tier=str(raw["tier"]),
                fixture_id=str(raw["fixture_id"]),
                query=str(raw["query"]),
                env={str(k): v for k, v in env.items()},
                expected_paths=[str(x) for x in (raw.get("expected_paths") or [])],
                required=bool(raw.get("required", False)),
                tags=[str(t) for t in tags],
                stability_hash=stability,
            )
        )
    return out


def _fixture_dir(repo_root: Path, fixture_id: str) -> Path:
    if fixture_id.startswith("external:"):
        raise ValueError(
            "external fixtures require TOKEN_REDUCER_BENCHMARK_FETCH (not implemented in this runner)"
        )
    return (repo_root / "benchmarks" / "fixtures" / fixture_id).resolve()


def _seed_fixture(conn: sqlite3.Connection, fixture_path: Path) -> int:
    n = 0
    for py in sorted(fixture_path.rglob("*.py")):
        if py.name.startswith("."):
            continue
        text = py.read_text(encoding="utf-8")
        upsert_document(
            conn,
            str(py.resolve()),
            raw_text=text,
            cleaned_text=text,
            chunk_size_words=24,
            overlap_words=0,
            dimensions=64,
            embedding_backend="hash",
            embedding_model=None,
        )
        n += 1
    conn.commit()
    return n


def _paths_hit(
    workspace_root: Path | None, selected_sources: list[str], expected: list[str]
) -> tuple[bool, list[str]]:
    missing: list[str] = []
    norm_selected: list[str] = []
    for s in selected_sources:
        p = Path(s).resolve()
        if workspace_root is not None:
            wr = workspace_root.resolve()
            try:
                rel = p.relative_to(wr).as_posix().replace("\\", "/")
                norm_selected.append(rel)
                continue
            except ValueError:
                pass
        norm_selected.append(p.as_posix().replace("\\", "/"))

    for exp in expected:
        exp_n = exp.replace("\\", "/").lstrip("./")
        hit = any(
            ns.endswith(exp_n) or ns == exp_n or Path(ns).name == exp_n for ns in norm_selected
        )
        if not hit:
            missing.append(exp)
    return len(missing) == 0, missing


def _recall_at_k(expected: list[str], hits: int) -> float:
    if not expected:
        return 1.0
    return max(0.0, min(1.0, hits / len(expected)))


def build_decision_trace(state: ContextRunState) -> dict[str, Any]:
    route = state.execution_route
    tier = route.tier if route is not None else "complex"
    skill: str | None = None
    if route is not None and route.tier == "tool":
        skill = route.skill_id
    sub_meta = state.subagent_debug if isinstance(state.subagent_debug, dict) else {}
    ran = sub_meta.get("subagents")
    subagents_used = list(ran) if isinstance(ran, list) else []
    compression_triggered = bool(state.bullets)
    return {
        "tier": tier,
        "skill_selected": skill,
        "subagents_used": subagents_used,
        "compression_triggered": compression_triggered,
    }


@contextlib.contextmanager
def _env_patches(env: dict[str, Any]) -> Iterator[None]:
    stack: list[Any] = []
    tier_val = env.get("infer_retrieval_tier")
    if isinstance(tier_val, str) and tier_val:
        stack.append(
            patch("token_reducer.orchestrator.infer_retrieval_tier", return_value=tier_val)
        )
    for p in stack:
        p.start()
    try:
        yield
    finally:
        for p in reversed(stack):
            p.stop()


def run_scenario(
    scenario: BenchmarkScenario,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    try:
        fixture_path = _fixture_dir(repo_root, scenario.fixture_id)
    except ValueError as e:
        return {
            "schema_version": SCHEMA_VERSION,
            "git_sha": _repo_git_sha(repo_root),
            "scenario_id": scenario.scenario_id,
            "tier": scenario.tier,
            "required": scenario.required,
            "tags": scenario.tags,
            "skipped": "network_disabled",
            "note": str(e),
            "deterministic": {
                "pass": False,
                "missing_expected_paths": list(scenario.expected_paths),
            },
        }

    if not fixture_path.is_dir():
        return {
            "schema_version": SCHEMA_VERSION,
            "git_sha": _repo_git_sha(repo_root),
            "scenario_id": scenario.scenario_id,
            "tier": scenario.tier,
            "required": scenario.required,
            "tags": scenario.tags,
            "skipped": "fixture_missing",
            "error": f"fixture not found: {fixture_path}",
        }

    timings_ms: dict[str, float] = {}
    t_wall0 = time.perf_counter()

    fd, db_path_str = tempfile.mkstemp(prefix="bench_", suffix=".db")
    os.close(fd)
    db_path = Path(db_path_str)
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": _repo_git_sha(repo_root),
        "scenario_id": scenario.scenario_id,
        "tier": scenario.tier,
        "required": scenario.required,
        "tags": scenario.tags,
        "fixture_id": scenario.fixture_id,
        "timing_ms": timings_ms,
        "tokens": {},
        "retrieval": {},
        "decision_trace": {},
        "deterministic": {},
        "regression_flags": {"baseline_loaded": False},
    }
    try:
        conn = connect_db(db_path)
        try:
            n_docs = _seed_fixture(conn, fixture_path)
            row["fixture_documents_indexed"] = n_docs
            rt = get_runtime_defaults()
            intent = analyze_query_intent(scenario.query)
            si = detect_intent(scenario.query)
            route = resolve_execution_route(scenario.query, si)
            plan = build_routing_plan(
                route, structured_intent_to_dict(si), workspace_root=fixture_path
            )
            state = ContextRunState(
                conn=conn,
                db_path=db_path,
                query=scenario.query,
                intent=intent,
                runtime=rt,
                memory_blob={},
                session_id=f"bench-{scenario.scenario_id}",
                top_k=6,
                fts_k=8,
                vector_k=0,
                min_fts_hits=3,
                hybrid_mode="fallback",
                embedding_backend="hash",
                embedding_model=None,
                dimensions=64,
                word_budget=200,
                relevance_floor=0.12,
                workspace_root=fixture_path,
                execution_route=route,
                routing_plan=plan,
            )
            orch = ContextPipelineOrchestrator(state)

            with _env_patches(scenario.env):
                t0 = time.perf_counter()
                CostOptimizerAgent().run_prepare(orch)
                timings_ms["cost_prepare_ms"] = round((time.perf_counter() - t0) * 1000, 3)

                t0 = time.perf_counter()
                RetrieverAgent().run(orch)
                timings_ms["retrieve_ms"] = round((time.perf_counter() - t0) * 1000, 3)

                t0 = time.perf_counter()
                ReasoningEnhancerAgent().run(orch)
                timings_ms["reasoning_ms"] = round((time.perf_counter() - t0) * 1000, 3)

                t0 = time.perf_counter()
                CompressorAgent().run(orch)
                timings_ms["compress_ms"] = round((time.perf_counter() - t0) * 1000, 3)

            timings_ms["total_ms"] = round((time.perf_counter() - t_wall0) * 1000, 3)

            selected_sources = [c.source for c in state.selected]
            ok_paths, missing = _paths_hit(fixture_path, selected_sources, scenario.expected_paths)
            hits = len(scenario.expected_paths) - len(missing)
            recall = _recall_at_k(scenario.expected_paths, hits)

            row["timing_ms"] = timings_ms
            row["tokens"] = {
                "selected_chunk_estimate": sum(int(c.token_estimate) for c in state.selected),
                "final_bullet_estimate": sum(estimate_tokens(b) for b in state.bullets),
            }
            row["retrieval"] = {
                "fts_hits": len(state.fts_hits),
                "vector_hits": len(state.vector_hits),
                "retrieval_retry_done": state.retrieval_retry_done,
                "vector_retrieval_path": state.vector_retrieval_path,
                "scored_pool_size": len(state.scored_pool),
            }
            row["decision_trace"] = build_decision_trace(state)
            row["deterministic"] = {
                "pass": ok_paths,
                "missing_expected_paths": missing,
                "recall_at_expected": round(recall, 4),
                "selected_sources": selected_sources,
            }

            if scenario.stability_hash:
                sh = scenario.stability_hash
                algo_parts: list[str] = ["sha256"]
                inputs_desc: dict[str, Any] = {}
                digests: dict[str, str] = {}
                if sh.get("selected_sources", True):
                    digests["selected_sources"] = canonical_selected_sources_hash(
                        selected_sources, workspace_root=fixture_path
                    )
                    inputs_desc["selected_sources"] = True
                if state.bullets and sh.get("bullets", True):
                    order = sh.get("bullet_order", "as_is")
                    collapse = bool(sh.get("collapse_ws", False))
                    if order not in ("as_is", "sorted"):
                        order = "as_is"
                    digests["bullets"] = canonical_bullets_hash(
                        state.bullets, bullet_order=order, collapse_ws=collapse
                    )
                    inputs_desc["bullets"] = {"bullet_order": order, "collapse_ws": collapse}
                keys = sh.get("plugin_keys")
                if (
                    isinstance(keys, list)
                    and state.claude_context
                    and isinstance(state.claude_context, dict)
                ):
                    ks = [str(k) for k in keys]
                    digests["plugin_subset"] = plugin_subset_digest(state.claude_context, ks)
                    inputs_desc["plugin_keys"] = ks
                row["stability_hash_algorithm"] = "+".join(algo_parts)
                row["stability_hash_inputs"] = inputs_desc
                row["stability_digest"] = digests
        finally:
            conn.close()
    except Exception as ex:
        row["error"] = str(ex)
        row["deterministic"] = {
            "pass": False,
            "missing_expected_paths": list(scenario.expected_paths),
        }
    finally:
        db_path.unlink(missing_ok=True)

    return row


def _tier_matches(run_tier: str, scenario_tier: str) -> bool:
    if run_tier == "smoke":
        return scenario_tier == "smoke"
    if run_tier == "nightly":
        return scenario_tier in ("smoke", "nightly")
    if run_tier == "weekly":
        return scenario_tier in ("smoke", "nightly", "weekly")
    return scenario_tier == run_tier


def run_suite(
    *,
    tier: str,
    repo_root: Path,
    scenarios_dir: Path | None = None,
    fail_fast: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    sd = scenarios_dir or (repo_root / "benchmarks" / "scenarios")
    scenarios = load_scenarios(sd)
    rows: list[dict[str, Any]] = []
    failed_required = False
    for sc in scenarios:
        if not _tier_matches(tier, sc.tier):
            continue
        row = run_scenario(sc, repo_root=repo_root)
        rows.append(row)
        raw_det = row.get("deterministic")
        det: dict[str, Any] = raw_det if isinstance(raw_det, dict) else {}
        passed = bool(det.get("pass", False)) and "error" not in row
        if sc.required and not passed:
            failed_required = True
            if fail_fast:
                break
    return rows, failed_required
