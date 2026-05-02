"""Benchmark harness unit + smoke integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from token_reducer.benchmark.runner import (
    SCHEMA_VERSION,
    build_decision_trace,
    load_scenarios,
    run_scenario,
    run_suite,
)
from token_reducer.benchmark.stability import (
    canonical_bullets_hash,
    canonical_selected_sources_hash,
)
from token_reducer.execution_route import ExecutionRoute
from token_reducer.orchestrator import ContextRunState


def test_load_scenarios_reads_json(tmp_path: Path) -> None:
    d = tmp_path / "scenarios"
    d.mkdir()
    (d / "a.json").write_text(
        json.dumps(
            {
                "scenario_id": "x",
                "tier": "smoke",
                "fixture_id": "micro_py",
                "query": "q",
                "env": {},
                "expected_paths": [],
                "required": False,
                "tags": ["t"],
            }
        ),
        encoding="utf-8",
    )
    scenarios = load_scenarios(d)
    assert len(scenarios) == 1
    assert scenarios[0].scenario_id == "x"


def test_stability_hashes_deterministic(tmp_path: Path) -> None:
    (tmp_path / "lib").mkdir()
    (tmp_path / "svc.py").write_text("x", encoding="utf-8")
    (tmp_path / "lib" / "a.py").write_text("y", encoding="utf-8")
    p1 = tmp_path / "svc.py"
    p2 = tmp_path / "lib" / "a.py"
    h1 = canonical_selected_sources_hash([str(p1), str(p2)], workspace_root=tmp_path)
    h2 = canonical_selected_sources_hash([str(p2), str(p1)], workspace_root=tmp_path)
    assert h1 == h2
    b1 = canonical_bullets_hash([" a ", "b"], bullet_order="sorted", collapse_ws=True)
    b2 = canonical_bullets_hash(["b", "  a  "], bullet_order="sorted", collapse_ws=True)
    assert b1 == b2


def test_build_decision_trace_shape() -> None:
    st = cast(
        ContextRunState,
        SimpleNamespace(
            execution_route=ExecutionRoute(
                tier="tool",
                skill_id="doc-coauthoring",
                skip_subagents=False,
                skip_lsp=False,
                retrieval_scale=1.0,
                reducer_token_threshold=2800,
                efficiency_score=1.0,
            ),
            subagent_debug={"subagents": ["filter", "ranking"]},
            bullets=["x"],
        ),
    )
    dt = build_decision_trace(st)
    assert dt["tier"] == "tool"
    assert dt["skill_selected"] == "doc-coauthoring"
    assert dt["subagents_used"] == ["filter", "ranking"]
    assert dt["compression_triggered"] is True


def test_smoke_jwt_micro_scenario_passes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scenarios_dir = repo_root / "benchmarks" / "scenarios"
    assert scenarios_dir.is_dir()
    scenarios = load_scenarios(scenarios_dir)
    target = next((s for s in scenarios if s.scenario_id == "smoke_jwt_micro"), None)
    assert target is not None
    row = run_scenario(target, repo_root=repo_root)
    assert row["schema_version"] == SCHEMA_VERSION
    assert row.get("skipped") is None
    det = row["deterministic"]
    assert det["pass"] is True
    assert "decision_trace" in row
    assert row["decision_trace"]["tier"] in ("simple", "tool", "complex")


def test_run_suite_smoke_tier() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rows, failed = run_suite(tier="smoke", repo_root=repo_root)
    assert not failed
    assert any(r.get("scenario_id") == "smoke_jwt_micro" for r in rows)
