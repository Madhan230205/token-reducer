from __future__ import annotations

import json

from token_reducer.feedback import (
    load_workspace_prune_ema,
    persist_workspace_prune_ema,
    read_strategy_prune_adjustments,
)


def test_persist_workspace_prune_ema_roundtrip(tmp_path) -> None:
    ws = tmp_path / "proj"
    ws.mkdir()
    persist_workspace_prune_ema(ws, {"failure_adjacent": -1})
    ema = load_workspace_prune_ema(ws)
    assert "failure_adjacent" in ema
    persist_workspace_prune_ema(ws, {"failure_adjacent": 1})
    ema2 = load_workspace_prune_ema(ws)
    assert ema2["failure_adjacent"] != ema["failure_adjacent"]


def test_read_strategy_merges_workspace_bias(tmp_path) -> None:
    ws = tmp_path / "repo"
    ws.mkdir()
    p = tmp_path / "fb.jsonl"
    sid = "behavior_truth"
    p.write_text("\n".join(json.dumps({"context_words": 300, "extra": {"strategy_id": sid}}) for _ in range(5)) + "\n")
    persist_workspace_prune_ema(ws, {sid: -1})
    merged = read_strategy_prune_adjustments(p, workspace_root=ws, min_samples=4, fat_words=200, lean_words=72)
    assert merged.get(sid) == -1
