"""MCP server engine: workspace indexing, search, slicing, tool registration (hash backend)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from token_reducer import mcp_server
from token_reducer.mcp_server import Engine, _fmt_search, build_server, db_path_for

PY_FILE = '''"""Payments module."""

import hashlib

TAX_RATE = 0.2


def compute_invoice_total(items: list[float], discount: float = 0.0) -> float:
    """Sum line items, apply discount, then tax."""
    subtotal = sum(items)
    discounted = subtotal * (1.0 - discount)
    return discounted * (1.0 + TAX_RATE)


class RefundPolicy:
    """Decides whether a refund is allowed."""

    def __init__(self, window_days: int) -> None:
        self.window_days = window_days

    def allows(self, age_days: int) -> bool:
        return age_days <= self.window_days
'''

MD_FILE = """# Billing guide

Invoices are totalled with a discount and then tax is applied.
Refunds are allowed inside the refund window.
"""


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ws"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "billing.py").write_text(PY_FILE, encoding="utf-8")
    (root / "docs.md").write_text(MD_FILE, encoding="utf-8")
    (root / "notes.txt").write_text(
        "café naïve 日本語 — unicode must not crash indexing\n" * 5, encoding="utf-8"
    )
    monkeypatch.setenv("TOKEN_REDUCER_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("TOKEN_REDUCER_DB", str(tmp_path / "idx" / "index.db"))
    return root


@pytest.fixture
def engine(workspace: Path) -> Engine:
    return Engine(root=workspace, background_embeddings=False)


class TestIndexing:
    def test_sync_indexes_workspace_files(self, engine: Engine) -> None:
        stats = engine.sync(force=True)
        assert stats["indexed_files"] == 3
        assert stats["chunks"] >= 3
        assert engine.status()["documents"] == 3

    def test_second_sync_is_incremental(self, engine: Engine, workspace: Path) -> None:
        engine.sync(force=True)
        assert engine.sync(force=True)["indexed_files"] == 0
        (workspace / "pkg" / "new.py").write_text(
            "def brand_new_symbol():\n    return 1\n", encoding="utf-8"
        )
        assert engine.sync(force=True)["indexed_files"] == 1

    def test_sync_is_throttled(self, engine: Engine) -> None:
        first = engine.sync(force=True)
        assert engine.sync() is first

    def test_db_path_is_outside_workspace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TOKEN_REDUCER_DB", raising=False)
        p = db_path_for(tmp_path)
        assert tmp_path not in p.parents
        assert p.name == "index.db"


class TestSearch:
    def test_search_finds_function_by_description(self, engine: Engine) -> None:
        hits = engine.search("invoice total with discount and tax", top_k=3)
        assert hits
        assert hits[0]["file"] == "pkg/billing.py"
        assert hits[0]["symbol"] == "compute_invoice_total"

    def test_code_and_prose_both_surface_with_code_boosted(self, engine: Engine) -> None:
        hits = engine.search("refund window allowed", top_k=3)
        files = [h["file"] for h in hits]
        assert "pkg/billing.py" in files[:2] and "docs.md" in files[:2]
        code = next(h for h in hits if h["file"] == "pkg/billing.py")
        assert "RefundPolicy" in code["preview"] or code["symbol"] == "RefundPolicy"

    def test_find_symbol(self, engine: Engine) -> None:
        occ = engine.symbol("RefundPolicy", top_k=5)
        assert occ and occ[0]["file"] == "pkg/billing.py"
        assert "RefundPolicy" in occ[0]["snippet"]

    def test_fmt_search_is_compact_and_informative(self, engine: Engine) -> None:
        text = _fmt_search(engine.search("compute invoice", top_k=2))
        assert "pkg/billing.py" in text
        assert "[function compute_invoice_total]" in text
        assert len(text) < 2000

    def test_no_hits_message(self, engine: Engine) -> None:
        engine.sync(force=True)
        assert "No matches" in _fmt_search(engine.search("zzqx nonexistent", top_k=3))

    def test_context_returns_relative_files(self, engine: Engine) -> None:
        ctx = engine.context("how is the invoice total computed", top_k=2)
        assert "summary" in ctx
        assert all(not Path(f).is_absolute() for f in ctx["relevant_files"])
        assert "tokens" in ctx and "decision_trace" not in ctx


class TestReadSlice:
    def test_reads_line_range(self, engine: Engine) -> None:
        res = engine.read_slice("pkg/billing.py", 5, 6)
        assert res["ok"] and res["start"] == 5 and res["end"] == 6
        assert "TAX_RATE" in res["content"]

    def test_rejects_paths_outside_workspace(self, engine: Engine, tmp_path: Path) -> None:
        outside = tmp_path / "secret.txt"
        outside.write_text("x", encoding="utf-8")
        assert engine.read_slice(str(outside), 1, 1)["ok"] is False
        assert engine.read_slice("../secret.txt", 1, 1)["ok"] is False

    def test_end_zero_means_eof(self, engine: Engine) -> None:
        res = engine.read_slice("docs.md", 1, 0)
        assert res["end"] == res["total_lines"]


class TestServer:
    def test_registers_expected_tools(self, engine: Engine) -> None:
        server = build_server(engine)
        names = {t.name for t in asyncio.run(server.list_tools())}
        assert names == {
            "search_code",
            "get_context",
            "find_symbol",
            "read_file_slice",
            "index_status",
            "reindex",
        }

    def test_tool_call_roundtrip(self, engine: Engine) -> None:
        server = build_server(engine)
        result = asyncio.run(
            server.call_tool("search_code", {"query": "refund policy", "top_k": 2})
        )
        content = result[0] if isinstance(result, tuple) else result
        text = "".join(getattr(c, "text", "") for c in content)
        assert "RefundPolicy" in text


def test_background_upgrade_is_noop_for_hash_backend(engine: Engine) -> None:
    engine.sync(force=True)
    assert engine.upgrade_embeddings() == 0
    assert mcp_server.EMBED_BATCH > 0
