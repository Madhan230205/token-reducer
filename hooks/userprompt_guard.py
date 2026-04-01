#!/usr/bin/env python3
"""User prompt guardrails for token-reducer.

Goals:
1) Warn when user appears to paste very large raw content (pipeline bypass risk).
2) Periodically remind to compact/start fresh sessions to avoid context drift.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


MAX_PROMPT_WORDS = 900
MAX_PROMPT_LINES = 120
REMINDER_TURNS = {12, 20, 28, 36}


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def extract_prompt(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("user_prompt", "prompt", "message", "input"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        for value in payload.values():
            extracted = extract_prompt(value)
            if extracted:
                return extracted
        return ""

    if isinstance(payload, list):
        for item in payload:
            extracted = extract_prompt(item)
            if extracted:
                return extracted
        return ""

    return payload if isinstance(payload, str) else ""


def extract_session_id(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("session_id", "sessionId", "conversation_id", "conversationId"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            sid = extract_session_id(value)
            if sid:
                return sid
    elif isinstance(payload, list):
        for item in payload:
            sid = extract_session_id(item)
            if sid:
                return sid
    return "default"


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"sessions": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"sessions": {}}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        print(json.dumps({}), file=sys.stdout)
        return 0

    prompt = extract_prompt(payload)
    session_id = extract_session_id(payload)

    plugin_root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    state_path = Path(plugin_root) / ".cache" / "token-reducer" / "prompt_guard_state.json"
    state = load_state(state_path)
    sessions = state.setdefault("sessions", {})
    turns = int(sessions.get(session_id, 0)) + 1
    sessions[session_id] = turns
    state["sessions"] = sessions
    try:
        save_state(state_path, state)
    except Exception:
        pass

    messages: list[str] = []
    if prompt:
        word_count = len(prompt.split())
        line_count = prompt.count("\n") + 1
        has_compact_packet = "CONTEXT_PACKET_START" in prompt
        has_token_reducer_intent = "/token-reducer" in prompt or "token-reducer" in prompt.lower()

        if (
            (word_count > MAX_PROMPT_WORDS or line_count > MAX_PROMPT_LINES)
            and not has_compact_packet
            and not has_token_reducer_intent
        ):
            messages.append(
                "⚠️ Large raw prompt detected. This may bypass token reduction and burn tokens. "
                "Prefer: keep prompt short, pass files/logs as inputs, then run /token-reducer first. "
                f"(approx_tokens={estimate_tokens(prompt)})"
            )

    if turns in REMINDER_TURNS:
        messages.append(
            "🧹 Session hygiene reminder: context history is growing. Run /compact at milestones and start a fresh chat when switching major tasks."
        )

    print(json.dumps({"systemMessage": "\n\n".join(messages)}) if messages else json.dumps({}), file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
