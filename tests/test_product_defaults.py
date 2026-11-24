from __future__ import annotations

from pathlib import Path

from token_reducer.product_defaults import apply_plugin_product_defaults


def test_apply_skips_without_plugin_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CLAUDE_PLUGIN_ROOT", raising=False)
    monkeypatch.delenv("TOKEN_REDUCER_FEEDBACK", raising=False)
    monkeypatch.delenv("TOKEN_REDUCER_ADAPTATION_BRIEF", raising=False)
    apply_plugin_product_defaults(db_path=tmp_path / "x.db", session_id="default")
    import os

    assert os.environ.get("TOKEN_REDUCER_FEEDBACK") is None


def test_apply_skips_memory_db(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
    monkeypatch.delenv("TOKEN_REDUCER_FEEDBACK", raising=False)
    monkeypatch.delenv("TOKEN_REDUCER_ADAPTATION_BRIEF", raising=False)
    apply_plugin_product_defaults(db_path=Path(":memory:"), session_id="default")
    import os

    assert os.environ.get("TOKEN_REDUCER_FEEDBACK") is None


def test_apply_skips_session_auto(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
    monkeypatch.delenv("TOKEN_REDUCER_FEEDBACK", raising=False)
    db = tmp_path / "idx.db"
    db.write_bytes(b"")
    apply_plugin_product_defaults(db_path=db, session_id="auto")
    import os

    assert os.environ.get("TOKEN_REDUCER_FEEDBACK") is None


def test_apply_sets_when_plugin_and_disk_db(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
    monkeypatch.delenv("TOKEN_REDUCER_FEEDBACK", raising=False)
    monkeypatch.delenv("TOKEN_REDUCER_ADAPTATION_BRIEF", raising=False)
    db = tmp_path / "idx.db"
    db.write_bytes(b"")
    apply_plugin_product_defaults(db_path=db, session_id="default")
    import os

    assert os.environ["TOKEN_REDUCER_FEEDBACK"] == "1"
    assert os.environ["TOKEN_REDUCER_ADAPTATION_BRIEF"] == "1"


def test_apply_respects_minimal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
    monkeypatch.setenv("TOKEN_REDUCER_MINIMAL", "1")
    monkeypatch.delenv("TOKEN_REDUCER_FEEDBACK", raising=False)
    db = tmp_path / "idx.db"
    db.write_bytes(b"")
    apply_plugin_product_defaults(db_path=db, session_id="default")
    import os

    assert os.environ.get("TOKEN_REDUCER_FEEDBACK") is None


def test_apply_preserves_explicit_feedback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path))
    monkeypatch.setenv("TOKEN_REDUCER_FEEDBACK", "0")
    monkeypatch.delenv("TOKEN_REDUCER_ADAPTATION_BRIEF", raising=False)
    db = tmp_path / "idx.db"
    db.write_bytes(b"")
    apply_plugin_product_defaults(db_path=db, session_id="default")
    import os

    assert os.environ["TOKEN_REDUCER_FEEDBACK"] == "0"
    assert os.environ["TOKEN_REDUCER_ADAPTATION_BRIEF"] == "1"
