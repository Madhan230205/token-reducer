"""Shadow linter: run a quick syntax/build check on a file after edits."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


def _shell_quote_path_for_template(path: Path) -> str:
    """Quote a file path for embedding in a shell string after ``{file}`` substitution.

    On Windows (``cmd.exe``), ``shlex.quote`` uses single quotes, which ``cmd`` does not
    treat as argument delimiters; use ``subprocess.list2cmdline`` instead. On POSIX,
    ``shlex.quote`` is correct.
    """
    resolved = str(path.resolve())
    if sys.platform == "win32":
        return subprocess.list2cmdline([resolved])
    return shlex.quote(resolved)


def _normalize_ext(ext: str) -> str:
    e = ext.strip().lower()
    if not e:
        return ""
    return e if e.startswith(".") else f".{e}"


def run_shadow_linter(
    file_path: Path,
    ext: str,
    cmds: dict[str, str],
    timeout: int,
    timeouts_by_ext: dict[str, int] | None = None,
) -> tuple[bool, str]:
    """Run the configured command for this extension; return (ok, message_or_logs).

    ``timeout`` is the default ceiling in seconds. ``timeouts_by_ext`` maps
    extensions (e.g. ``.rs``) to longer limits for heavy compilers.
    """
    key = _normalize_ext(ext)
    template = cmds.get(key)
    if not template:
        return True, "No linter configured"

    abs_path = _shell_quote_path_for_template(file_path)
    cmd = template.replace("{file}", abs_path)

    effective_timeout = timeout
    if timeouts_by_ext:
        override = timeouts_by_ext.get(key)
        if isinstance(override, int) and override >= 1:
            effective_timeout = override

    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "Linter timed out"

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()

    if proc.returncode == 0:
        return True, "Lint passed"

    logs = "\n".join(part for part in (out, err) if part)
    return False, logs if logs else f"exit code {proc.returncode}"
