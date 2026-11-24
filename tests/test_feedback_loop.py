from __future__ import annotations

import json
from pathlib import Path

from token_reducer.feedback import (
    FeedbackLoopAdjustments,
    build_adaptation_brief,
    feedback_loop_adjustments,
)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_feedback_loop_neutral_without_history(tmp_path: Path) -> None:
    adj = feedback_loop_adjustments(log_path=tmp_path / "missing.jsonl")
    assert adj == FeedbackLoopAdjustments(1.0, 0.0)


def test_feedback_loop_tightens_when_recent_packets_fat(tmp_path: Path) -> None:
    p = tmp_path / "fb.jsonl"
    fat = [{"context_words": 220, "extra": {}} for _ in range(8)]
    _write_rows(p, fat)
    adj = feedback_loop_adjustments(log_path=p, tail_lines=40, min_samples=6)
    assert adj.retrieval_scale_mult < 1.0
    assert adj.relevance_floor_delta > 0


def test_feedback_loop_loosens_when_recent_packets_lean(tmp_path: Path) -> None:
    p = tmp_path / "fb.jsonl"
    lean = [{"context_words": 35, "extra": {}} for _ in range(8)]
    _write_rows(p, lean)
    adj = feedback_loop_adjustments(log_path=p, tail_lines=40, min_samples=6)
    assert adj.retrieval_scale_mult > 1.0
    assert adj.relevance_floor_delta < 0


def test_feedback_loop_bad_outcomes_extra_tighten(tmp_path: Path) -> None:
    p = tmp_path / "fb.jsonl"
    rows = [{"context_words": 120, "extra": {"outcome": "bad"}} for _ in range(6)]
    _write_rows(p, rows)
    adj = feedback_loop_adjustments(log_path=p, tail_lines=20, min_samples=6)
    assert adj.retrieval_scale_mult < 1.0


def test_build_adaptation_brief_covers_model_and_feedback(tmp_path: Path) -> None:
    adj = feedback_loop_adjustments(log_path=tmp_path / "missing.jsonl")
    lines = build_adaptation_brief(
        adj=adj,
        model="haiku-3",
        feedback_source_boosts={},
        subagent_meta=None,
        compression_level_effective="high",
        execution_tier="complex",
    )
    assert any("Haiku" in ln for ln in lines)
    assert any("complex" in ln for ln in lines)
    assert any("high" in ln for ln in lines)

    adj2 = FeedbackLoopAdjustments(0.9, 0.02)
    lines2 = build_adaptation_brief(
        adj=adj2,
        model=None,
        feedback_source_boosts={"foo.py": 0.04},
        subagent_meta={"tokens_saved": 120, "specialization": {"legacy_intent": "bug_fix"}},
        compression_level_effective="medium",
    )
    assert any("Feedback loop" in ln for ln in lines2)
    assert any("foo.py" in ln for ln in lines2)
    assert any("120" in ln for ln in lines2)
    assert any("bug-fix" in ln.lower() or "coordination" in ln.lower() for ln in lines2)
