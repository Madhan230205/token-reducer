"""End-to-end CLI integration tests for token-reducer.

Invokes `python scripts/context_pipeline.py <subcommand>` via subprocess and
asserts on exit codes and output content.  Each test uses an isolated temp DB
so runs do not share state.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Project root is one level above this file's directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CLI = [sys.executable, str(PROJECT_ROOT / "scripts" / "context_pipeline.py")]
TIMEOUT = 120  # seconds — generous for first-run model/index build


def _run(*args: str) -> subprocess.CompletedProcess:
    """Run the CLI with the given args and return the CompletedProcess."""
    cmd = CLI + list(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT,
        cwd=str(PROJECT_ROOT),
    )


# ---------------------------------------------------------------------------
# benchmark command
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_exit_code_zero(self, tmp_path: Path) -> None:
        db = str(tmp_path / "bench.db")
        result = _run("benchmark", "--inputs", str(PROJECT_ROOT), "--db", db)
        assert result.returncode == 0, (
            f"benchmark exited {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_banner_on_stderr(self, tmp_path: Path) -> None:
        db = str(tmp_path / "bench.db")
        result = _run("benchmark", "--inputs", str(PROJECT_ROOT), "--db", db)
        # The human-readable summary is written to stderr via rich Console(stderr=True)
        assert "TOKEN REDUCER BENCHMARK" in result.stderr, (
            f"Expected 'TOKEN REDUCER BENCHMARK' in stderr.\nSTDERR:\n{result.stderr}"
        )

    def test_token_savings_in_stderr(self, tmp_path: Path) -> None:
        db = str(tmp_path / "bench.db")
        result = _run("benchmark", "--inputs", str(PROJECT_ROOT), "--db", db)
        assert "Token savings:" in result.stderr, (
            f"Expected 'Token savings:' in stderr.\nSTDERR:\n{result.stderr}"
        )

    def test_files_indexed_positive(self, tmp_path: Path) -> None:
        db = str(tmp_path / "bench.db")
        result = _run("benchmark", "--inputs", str(PROJECT_ROOT), "--db", db)
        assert "Files indexed:" in result.stderr, (
            f"Expected 'Files indexed:' in stderr.\nSTDERR:\n{result.stderr}"
        )

    def test_stdout_is_valid_json(self, tmp_path: Path) -> None:
        db = str(tmp_path / "bench.db")
        result = _run("benchmark", "--inputs", str(PROJECT_ROOT), "--db", db)
        assert result.returncode == 0
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            pytest.fail(f"benchmark stdout is not valid JSON: {exc}\nSTDOUT:\n{result.stdout[:500]}")
        assert report["benchmark_summary"]["files_indexed"] > 0

    def test_no_files_exits_nonzero(self, tmp_path: Path) -> None:
        """Passing a directory with no indexable files should exit non-zero."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        db = str(tmp_path / "bench.db")
        result = _run("benchmark", "--inputs", str(empty_dir), "--db", db)
        assert result.returncode != 0, "Expected non-zero exit for empty inputs"


# ---------------------------------------------------------------------------
# run command (default human-readable output)
# ---------------------------------------------------------------------------


class TestRunHuman:
    def test_exit_code_zero(self, tmp_path: Path) -> None:
        db = str(tmp_path / "run.db")
        result = _run("run", "--inputs", str(PROJECT_ROOT), "--query", "error handling", "--db", db)
        assert result.returncode == 0, (
            f"run exited {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_output_is_not_json(self, tmp_path: Path) -> None:
        """Without --json flag the output should be the human-readable packet, not JSON."""
        db = str(tmp_path / "run.db")
        result = _run("run", "--inputs", str(PROJECT_ROOT), "--query", "error handling", "--db", db)
        assert result.returncode == 0
        # Human output should NOT be parseable as a dict with 'query' key at top level
        try:
            parsed = json.loads(result.stdout)
            # If it parses, it should NOT have the full model_dump structure
            is_full_model = isinstance(parsed, dict) and "selected_chunks" in parsed and "token_metrics" in parsed
            assert not is_full_model, "Expected human-readable output but got full JSON model dump"
        except json.JSONDecodeError:
            pass  # Non-JSON output is correct


# ---------------------------------------------------------------------------
# run command with --json flag
# ---------------------------------------------------------------------------


class TestRunJson:
    def test_exit_code_zero(self, tmp_path: Path) -> None:
        db = str(tmp_path / "run_json.db")
        result = _run(
            "run", "--inputs", str(PROJECT_ROOT), "--query", "error handling", "--db", db, "--json"
        )
        assert result.returncode == 0, (
            f"run --json exited {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_output_is_valid_json(self, tmp_path: Path) -> None:
        db = str(tmp_path / "run_json.db")
        result = _run(
            "run", "--inputs", str(PROJECT_ROOT), "--query", "error handling", "--db", db, "--json"
        )
        assert result.returncode == 0
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            pytest.fail(f"run --json stdout is not valid JSON: {exc}\nSTDOUT:\n{result.stdout[:500]}")
        assert isinstance(data, dict)

    def test_required_keys_present(self, tmp_path: Path) -> None:
        db = str(tmp_path / "run_json.db")
        result = _run(
            "run", "--inputs", str(PROJECT_ROOT), "--query", "error handling", "--db", db, "--json"
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        for key in ("query", "selected_chunks", "token_metrics", "retrieval", "packet"):
            assert key in data, f"Missing key '{key}' in JSON output. Keys: {list(data.keys())}"

    def test_selected_chunks_positive(self, tmp_path: Path) -> None:
        db = str(tmp_path / "run_json.db")
        result = _run(
            "run", "--inputs", str(PROJECT_ROOT), "--query", "error handling", "--db", db, "--json"
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert int(data["selected_chunks"]) > 0, (
            f"Expected selected_chunks > 0, got {data['selected_chunks']}"
        )

    def test_query_echoed_in_output(self, tmp_path: Path) -> None:
        db = str(tmp_path / "run_json.db")
        query = "error handling"
        result = _run("run", "--inputs", str(PROJECT_ROOT), "--query", query, "--db", db, "--json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["query"] == query


# ---------------------------------------------------------------------------
# index command
# ---------------------------------------------------------------------------


class TestIndex:
    def test_exit_code_zero(self, tmp_path: Path) -> None:
        db = str(tmp_path / "index.db")
        result = _run("index", "--inputs", str(PROJECT_ROOT), "--db", db)
        assert result.returncode == 0, (
            f"index exited {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_stdout_json_files_indexed(self, tmp_path: Path) -> None:
        db = str(tmp_path / "index.db")
        result = _run("index", "--inputs", str(PROJECT_ROOT), "--db", db)
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["files_indexed"] > 0
        assert "chunks_indexed" in data
        assert data["chunks_indexed"] > 0


# ---------------------------------------------------------------------------
# Error / edge-case handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_invalid_inputs_path_no_traceback(self, tmp_path: Path) -> None:
        """A nonexistent path should produce a graceful error, not a raw Python traceback."""
        nonexistent = str(tmp_path / "does_not_exist_xyz")
        db = str(tmp_path / "err.db")
        result = _run("run", "--inputs", nonexistent, "--query", "test", "--db", db)
        # Should not crash with unhandled exception — either non-zero exit or an error message
        combined = result.stdout + result.stderr
        assert "Traceback (most recent call last)" not in combined, (
            "CLI crashed with unhandled traceback on invalid inputs path"
        )
        # Expect either a non-zero exit code or an explicit error message
        has_error_msg = any(
            kw in combined.lower() for kw in ("no indexable", "error", "not found", "no files")
        )
        assert result.returncode != 0 or has_error_msg, (
            f"Expected non-zero exit or error message for invalid path.\n"
            f"Exit: {result.returncode}\nOutput: {combined[:400]}"
        )

    def test_run_missing_query_exits_nonzero(self, tmp_path: Path) -> None:
        """Omitting --query should cause a non-zero exit."""
        db = str(tmp_path / "err.db")
        result = _run("run", "--inputs", str(PROJECT_ROOT), "--db", db)
        assert result.returncode != 0, "Expected non-zero exit when --query is omitted"
