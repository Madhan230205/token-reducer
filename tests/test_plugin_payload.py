from __future__ import annotations

from token_reducer.models import Candidate
from token_reducer.plugin_output import build_claude_plugin_payload


def _c(chunk_id: int, source: str, text: str, idx: int = 0) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        source=source,
        chunk_index=idx,
        text=text,
        token_estimate=10,
    )


def test_plugin_payload_keeps_multiple_chunks_same_file_without_symbol_header() -> None:
    """Previously dedup used (file, symbol); stem fallback made all prose chunks one key."""
    path = "C:/proj/README.md"
    a = _c(1, path, "Section A about deployment.\nMore detail here.")
    b = _c(2, path, "Section B about security.\nOther lines.")
    payload = build_claude_plugin_payload("ops", "explain_code", [a, b])
    ctx = payload["code_context"]
    assert len(ctx) == 2
    assert {row["file"] for row in ctx} == {path}


def test_plugin_payload_skips_duplicate_chunk_id() -> None:
    c = _c(42, "x.py", "def foo():\n    return 1\n")
    payload = build_claude_plugin_payload("x", "explain_code", [c, c])
    assert len(payload["code_context"]) == 1


def test_plugin_context_strategy_folds_framing_into_summary() -> None:
    c = _c(1, "x.py", "def foo():\n    return 1\n")
    strat = {
        "strategy_id": "behavior_truth",
        "attention_frame": "Treat excerpts as the source of truth.",
        "merge_cap": 50,
        "prune_k": 14,
        "skip_fusion": False,
        "skip_neighborhood": False,
    }
    payload = build_claude_plugin_payload("q", "explain_code", [c], context_strategy=strat)
    assert "attention_frame" not in payload
    assert "context_shape" not in payload
    assert "Treat excerpts as the source of truth." in str(payload["summary"])


def test_plugin_patch_first_adds_guidance_and_smaller_code_cap() -> None:
    long_body = "def x():\n" + "    return 1\n" * 800
    c = _c(1, "big.py", long_body, 0)
    plain = build_claude_plugin_payload("q", "explain_code", [c], patch_first=False)
    patchy = build_claude_plugin_payload("q", "bug_fix", [c], patch_first=True)
    assert "edit_guidance" not in plain
    assert "edit_guidance" in patchy
    assert len(str(patchy["code_context"][0]["code"])) < len(str(plain["code_context"][0]["code"]))
