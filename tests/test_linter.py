"""Tests for shadow linter."""

from __future__ import annotations

from pathlib import Path
from subprocess import TimeoutExpired
from unittest.mock import patch

from token_reducer.linter import run_shadow_linter


def test_no_linter_for_unknown_extension(tmp_path: Path) -> None:
    p = tmp_path / "f.xyz"
    p.write_text("x", encoding="utf-8")
    ok, msg = run_shadow_linter(p, ".xyz", {".py": "python -m py_compile {file}"}, 5)
    assert ok and msg == "No linter configured"


def test_py_syntax_pass(tmp_path: Path) -> None:
    p = tmp_path / "ok.py"
    p.write_text("x = 1\n", encoding="utf-8")
    ok, msg = run_shadow_linter(p, ".py", {".py": "python -m py_compile {file}"}, 30)
    assert ok and msg == "Lint passed"


def test_path_with_space_in_directory(tmp_path: Path) -> None:
    d = tmp_path / "My Project"
    d.mkdir()
    p = d / "ok.py"
    p.write_text("x = 1\n", encoding="utf-8")
    ok, msg = run_shadow_linter(p, ".py", {".py": "python -m py_compile {file}"}, 30)
    assert ok and msg == "Lint passed"


def test_py_syntax_fail(tmp_path: Path) -> None:
    p = tmp_path / "bad.py"
    p.write_text("def x(\n", encoding="utf-8")
    ok, msg = run_shadow_linter(p, "py", {".py": "python -m py_compile {file}"}, 30)
    assert not ok
    assert "SyntaxError" in msg or "Error" in msg or msg


def test_timeout(tmp_path: Path) -> None:
    p = tmp_path / "slow.py"
    p.write_text("x=1\n", encoding="utf-8")
    with patch("token_reducer.linter.subprocess.run", side_effect=TimeoutExpired("cmd", 1)):
        ok, msg = run_shadow_linter(p, ".py", {".py": "python -m py_compile {file}"}, 1)
    assert not ok and msg == "Linter timed out"


def test_per_extension_timeout_override(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    p = tmp_path / "x.rs"
    p.write_text("fn main() {}\n", encoding="utf-8")
    mock_ret = MagicMock(returncode=0, stdout="", stderr="")
    with patch("token_reducer.linter.subprocess.run", return_value=mock_ret) as m:
        ok, msg = run_shadow_linter(
            p,
            ".rs",
            {".rs": "python -m py_compile {file}"},
            5,
            {".rs": 42},
        )
    assert ok and msg == "Lint passed"
    assert m.call_args is not None
    assert m.call_args.kwargs["timeout"] == 42
