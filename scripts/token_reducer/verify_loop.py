"""Deterministic verification hints after edits (lint-first; optional pytest path)."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .linter import run_shadow_linter


@dataclass(frozen=True)
class VerificationOutcome:
    success: bool
    errors: tuple[str, ...]
    failing_files: tuple[str, ...]


def build_verification_plan(
    task_mode: str,
    *,
    touched_files: list[str] | None = None,
    runtime_lint_cmds: dict[str, str] | None = None,
    lint_timeout: int = 45,
    query: str = "",
) -> dict[str, Any]:
    """Structured plan for post-patch checks — no network; callers run after apply_diff."""
    max_retries = 0
    if task_mode in ("debug", "refactor", "add_feature", "write_test"):
        max_retries = 2 if task_mode == "debug" else 1
    plan: dict[str, Any] = {
        "preferred_output": "patch",
        "max_retries": max_retries,
        "run_shadow_lint_on_touched": True,
        "hint": (
            "Prefer minimal SEARCH/REPLACE or AST-targeted hunks from apply_diff format; "
            "avoid full-file rewrites unless necessary."
        ),
    }
    if task_mode == "write_test":
        plan["suggested_test_command"] = "pytest -q --lf"
    elif task_mode in ("debug", "refactor"):
        plan["suggested_test_command"] = "pytest -q -x --maxfail=1"
    if touched_files:
        plan["touched_files"] = list(touched_files)[:32]
    if runtime_lint_cmds:
        plan["lint_commands_by_ext"] = dict(runtime_lint_cmds)
    plan["narrow_retry_hint"] = (
        "On verify failure: re-fetch only failing file chunks and error lines; "
        "do not expand full-repo context."
    )
    plan["lint_timeout_seconds"] = lint_timeout
    plan["failure_recovery"] = {
        "narrow_context_only": True,
        "max_secondary_retrieval_passes": 2 if task_mode == "debug" else 1,
        "requery_append_stderr_line": True,
        "focus_touched_files_first": True,
    }
    if query:
        plan["query_echo_for_retry"] = query[:200]
    return plan


def run_verification(
    touched_paths: list[Path],
    policy_dict: dict[str, Any],
    *,
    cmds: dict[str, str],
    timeout: int,
) -> VerificationOutcome:
    """Execute cheap checks: Python syntax (always for .py) + shadow linter when configured."""
    errors: list[str] = []
    failing: list[str] = []

    merged_cmds: dict[str, str] = dict(cmds)
    extra = policy_dict.get("lint_commands_by_ext")
    if isinstance(extra, dict):
        merged_cmds.update({str(k): str(v) for k, v in extra.items() if isinstance(v, str)})

    if not touched_paths:
        return VerificationOutcome(True, (), ())

    for p in touched_paths[:12]:
        if not p.is_file():
            errors.append(f"Missing file after patch: {p}")
            failing.append(str(p))
            continue
        suf = p.suffix.lower()
        if suf == ".py":
            try:
                ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                msg = f"{p.name}: {exc}"
                errors.append(msg)
                failing.append(str(p.resolve()))

        if merged_cmds:
            ok, lint_msg = run_shadow_linter(p, suf, merged_cmds, timeout, None)
            if not ok:
                errors.append(f"{p.name}: {lint_msg}")
                failing.append(str(p.resolve()))

    if (
        os.environ.get("TOKEN_REDUCER_AGENT_PYTEST", "").strip() in ("1", "true", "yes")
        and policy_dict.get("suggested_test_command")
        and touched_paths
    ):
        first = touched_paths[0]
        if first.is_file():
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", str(first), "--maxfail=1"],
                    capture_output=True,
                    text=True,
                    timeout=90,
                    cwd=str(first.parent),
                )
                if proc.returncode != 0:
                    tail = (proc.stdout or "")[-800:] + (proc.stderr or "")[-800:]
                    errors.append(f"pytest: {tail.strip() or proc.returncode}")
                    failing.append(str(first.resolve()))
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"pytest skipped: {exc}")

    ok = not errors
    return VerificationOutcome(ok, tuple(errors), tuple(dict.fromkeys(failing)))


def run_shadow_lint_on_files(
    paths: list[Path],
    *,
    cmds: dict[str, str],
    timeout: int,
) -> tuple[bool, str]:
    """Run configured shadow linter on the first few paths; cheap gate for verify loop."""
    if not paths:
        return True, "No files to lint"
    messages: list[str] = []
    for p in paths[:4]:
        if not p.is_file():
            continue
        ok, msg = run_shadow_linter(p, p.suffix, cmds, timeout)
        messages.append(f"{p.name}: {msg}")
        if not ok:
            return False, "\n".join(messages)
    return True, "\n".join(messages) if messages else "Lint skipped"
