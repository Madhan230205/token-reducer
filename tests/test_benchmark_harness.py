"""Benchmark harness unit + smoke integration tests."""

from __future__ import annotations

import json
from pathlib import Path

from token_reducer.audit_spine import (
    CompressionLossMetrics,
    CompressionSection,
    OutcomeSection,
    RetrievalCandidate,
    RetrievalDisagreement,
    RetrievalSection,
    RoutingSection,
    bootstrap_spine,
)
from token_reducer.benchmark.runner import (
    SCHEMA_VERSION,
    build_decision_trace_payload,
    load_scenarios,
    run_scenario,
    run_suite,
)
from token_reducer.benchmark.stability import (
    canonical_bullets_hash,
    canonical_selected_sources_hash,
)
from token_reducer.db import connect_db
from token_reducer.execution_route import ExecutionRoute
from token_reducer.intent import analyze_query_intent, detect_intent
from token_reducer.orchestrator import ContextRunState
from token_reducer.plugin_settings import get_runtime_defaults


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


def test_build_decision_trace_payload_shape(tmp_path: Path) -> None:
    db_path = tmp_path / "bench_unit.db"
    conn = connect_db(db_path)
    try:
        route = ExecutionRoute(
            tier="tool",
            skill_id="doc-coauthoring",
            skip_subagents=False,
            skip_lsp=False,
            retrieval_scale=1.0,
            reducer_token_threshold=2800,
            efficiency_score=1.0,
        )
        si = detect_intent("x")
        spine = bootstrap_spine("x", si, route)
        spine = spine.model_copy(
            update={
                "routing": RoutingSection(selected_tier="tool"),
                "retrieval": RetrievalSection(
                    candidates=(
                        RetrievalCandidate(
                            doc_id="a",
                            path="p.py",
                            chunk_id="99",
                            source="fts",
                            score=0.5,
                            rank=1,
                        ),
                    ),
                    disagreement=RetrievalDisagreement(),
                ),
                "compression": CompressionSection(
                    input_tokens=100,
                    output_tokens=50,
                    retained_chunks=("99",),
                    dropped_chunks=("100",),
                    loss_metrics=CompressionLossMetrics(
                        importance_retained_ratio=0.5,
                        salient_span_checksum="",
                    ),
                ),
                "outcome": OutcomeSection(latency_ms=10, final_confidence=0.8),
            }
        )
        rt = get_runtime_defaults()
        st = ContextRunState(
            conn=conn,
            db_path=db_path,
            query="x",
            intent=analyze_query_intent("x"),
            runtime=rt,
            memory_blob={},
            session_id="t",
            top_k=5,
            fts_k=8,
            vector_k=0,
            min_fts_hits=3,
            hybrid_mode="fallback",
            embedding_backend="hash",
            embedding_model=None,
            dimensions=64,
            word_budget=200,
            relevance_floor=0.1,
            workspace_root=None,
            execution_route=route,
            audit_spine=spine,
            subagent_debug={"subagents": ["filter", "ranking"]},
            bullets=["z"],
            omitted_redundant=[],
        )
        dt = build_decision_trace_payload(st)
        assert dt["schema_version"] == "decision_trace_v2"
        assert dt["routing_tier"] == "tool"
        assert dt["tool_skill_id"] == "doc-coauthoring"
        assert dt["subagents_used"] == ["filter", "ranking"]
        assert dt["compression_triggered"] is True
        assert "reliability" in dt
        assert len(dt["selected_chunks"]) >= 1
        assert dt["selected_chunks"][0]["chunk_id"] == "99"
    finally:
        conn.close()


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
    assert row["decision_trace"]["schema_version"] == "decision_trace_v2"
    assert row["decision_trace"]["routing_tier"] in ("simple", "tool", "complex")
    assert row.get("trace_bundle") is not None
    assert row["trace_bundle"]["trace_id"] == row["decision_trace"]["trace_id"]


def test_run_suite_smoke_tier() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rows, failed = run_suite(tier="smoke", repo_root=repo_root)
    assert not failed
    assert any(r.get("scenario_id") == "smoke_jwt_micro" for r in rows)
