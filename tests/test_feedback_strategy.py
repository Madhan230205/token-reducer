from __future__ import annotations

import json

from token_reducer.feedback import read_strategy_prune_adjustments


def _line(context_words: int, strategy_id: str) -> str:
    return json.dumps(
        {
            "ts": 1,
            "prompt_chars": 10,
            "context_words": context_words,
            "extra": {"strategy_id": strategy_id},
        }
    )


def test_read_strategy_prune_adjustments_fat_history_tightens(tmp_path) -> None:
    p = tmp_path / "fb.jsonl"
    sid = "failure_adjacent"
    p.write_text("\n".join(_line(300, sid) for _ in range(5)) + "\n", encoding="utf-8")
    adj = read_strategy_prune_adjustments(p, tail_lines=50, min_samples=4, fat_words=200, lean_words=72)
    assert adj.get(sid) == -1


def test_read_strategy_prune_adjustments_lean_history_widens(tmp_path) -> None:
    p = tmp_path / "fb.jsonl"
    sid = "behavior_truth"
    p.write_text("\n".join(_line(40, sid) for _ in range(5)) + "\n", encoding="utf-8")
    adj = read_strategy_prune_adjustments(p, tail_lines=50, min_samples=4, fat_words=200, lean_words=72)
    assert adj.get(sid) == 1


def test_read_strategy_prune_adjustments_too_few_samples(tmp_path) -> None:
    p = tmp_path / "fb.jsonl"
    p.write_text(_line(500, "x") + "\n" + _line(500, "x") + "\n", encoding="utf-8")
    adj = read_strategy_prune_adjustments(p, min_samples=4)
    assert adj == {}
