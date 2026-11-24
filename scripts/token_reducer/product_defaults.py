"""Plug-and-play defaults when running as the Claude Code plugin (``CLAUDE_PLUGIN_ROOT``).

The slash command should not require users to hunt for env vars. If the host sets
``CLAUDE_PLUGIN_ROOT`` and the run is a normal on-disk index (not stdin/ephemeral
compression), we turn on feedback logging + adaptation narrative automatically.

Opt out: ``TOKEN_REDUCER_MINIMAL=1`` (or ``true``/``yes``/``on``).
"""

from __future__ import annotations

import os
from pathlib import Path


def apply_plugin_product_defaults(*, db_path: Path, session_id: str) -> None:
    """Set sensible ``TOKEN_REDUCER_*`` defaults once per process when unset."""
    if not os.environ.get("CLAUDE_PLUGIN_ROOT", "").strip():
        return
    if os.environ.get("TOKEN_REDUCER_MINIMAL", "").strip().lower() in ("1", "true", "yes", "on"):
        return
    # Ephemeral stdin compression (`compress-raw`) — avoid JSONL spam / side effects
    if str(db_path) == ":memory:":
        return
    if session_id.strip().lower() == "auto":
        return

    if not os.environ.get("TOKEN_REDUCER_FEEDBACK"):
        os.environ["TOKEN_REDUCER_FEEDBACK"] = "1"
    if not os.environ.get("TOKEN_REDUCER_ADAPTATION_BRIEF"):
        os.environ["TOKEN_REDUCER_ADAPTATION_BRIEF"] = "1"
