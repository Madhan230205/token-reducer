#!/usr/bin/env python3
"""User prompt guardrails for token-reducer.

Goals:
1) Warn when user appears to paste very large raw content (pipeline bypass risk).
2) Periodically remind to compact/start fresh sessions to avoid context drift.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


MAX_PROMPT_WORDS = 900
MAX_PROMPT_LINES = 120
HARD_TRUNCATE_WORDS = 800
HARD_BLOCK_WORDS = 3000
REMINDER_TURNS_DEFAULT = {5, 8, 10, 12, 15, 20, 28, 36}
AUTO_COMPACT_TURN_DEFAULT = 10
AUTO_RESET_TURN_DEFAULT = 40
CRITICAL_RESET_TURN_DEFAULT = 50


def _load_guard_settings(plugin_root: str) -> dict[str, Any]:
    path = Path(plugin_root) / "settings.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    pg = data.get("promptGuard")
    tr = data.get("tokenReducer")
    out: dict[str, Any] = {}
    if isinstance(pg, dict):
        out["promptGuard"] = pg
    if isinstance(tr, dict):
        out["tokenReducer"] = tr
    return out


def _guard_params(plugin_root: str) -> tuple[int, int, int, set[int]]:
    blob = _load_guard_settings(plugin_root)
    pg = blob.get("promptGuard") or {}
    tr = blob.get("tokenReducer") or {}

    auto_compact = int(pg.get("autoCompactTurn", AUTO_COMPACT_TURN_DEFAULT))
    auto_reset = int(pg.get("autoResetTurn", AUTO_RESET_TURN_DEFAULT))
    critical = int(pg.get("criticalResetTurn", CRITICAL_RESET_TURN_DEFAULT))

    reminder: set[int] = set()
    if isinstance(pg.get("reminderTurns"), list):
        for x in pg["reminderTurns"]:
            try:
                reminder.add(int(x))
            except (TypeError, ValueError):
                pass
    hist = tr.get("historyCompactReminderTurns")
    if isinstance(hist, list):
        for x in hist:
            try:
                reminder.add(int(x))
            except (TypeError, ValueError):
                pass
    if not reminder:
        reminder = set(REMINDER_TURNS_DEFAULT)

    return auto_compact, auto_reset, critical, reminder


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

    auto_compact_turn, auto_reset_turn, critical_reset_turn, reminder_turns = _guard_params(
        plugin_root
    )

    messages: list[str] = []
    result: dict[str, Any] = {}

    if prompt:
        word_count = len(prompt.split())
        line_count = prompt.count("\n") + 1
        has_compact_packet = "CONTEXT_PACKET_START" in prompt
        has_token_reducer_intent = "/token-reducer" in prompt or "token-reducer" in prompt.lower()
        bypass = has_compact_packet or has_token_reducer_intent

        if not bypass and word_count > HARD_BLOCK_WORDS:
            # Hard block: reject the prompt entirely
            messages.append(
                f"🚫 Prompt BLOCKED: {word_count} words (~{estimate_tokens(prompt)} tokens) exceeds the "
                f"{HARD_BLOCK_WORDS}-word hard limit. Reduce your prompt size or pass large content "
                "via --inputs and run /token-reducer instead. Prompt was not submitted."
            )
            result["rejectInput"] = True
            result["systemMessage"] = "\n\n".join(messages)
            print(json.dumps(result), file=sys.stdout)
            return 0

        if not bypass and word_count > HARD_TRUNCATE_WORDS:
            # TPCH: Zero-Turn Auto-Compression — intercept before LLM sees the prompt
            messages.append(
                f"⚡ Auto-Compression Engaged: Intercepted {word_count} words "
                f"(~{estimate_tokens(prompt)} tokens) before sending to the AI. "
                "Compressing context to save tokens..."
            )
            try:
                pipeline_script = Path(plugin_root) / "scripts" / "context_pipeline.py"
                process = subprocess.Popen(
                    [sys.executable, str(pipeline_script), "compress-raw", "--word-budget", "600"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                compressed_packet, _ = process.communicate(input=prompt, timeout=30)
                if process.returncode == 0 and compressed_packet.strip():
                    saved = estimate_tokens(prompt) - estimate_tokens(compressed_packet)
                    messages.append(f"✅ Compressed: saved ~{saved} tokens on Turn 1.")
                    result["transformedPrompt"] = compressed_packet.strip()
                else:
                    raise RuntimeError("compression yielded empty output")
            except Exception:
                truncated_words = prompt.split()[:HARD_TRUNCATE_WORDS]
                result["transformedPrompt"] = " ".join(truncated_words)
                dropped = word_count - HARD_TRUNCATE_WORDS
                messages.append(
                    f"✂️ Prompt TRUNCATED: compression failed, dropped last {dropped} words."
                )

        elif not bypass and (word_count > MAX_PROMPT_WORDS or line_count > MAX_PROMPT_LINES):
            messages.append(
                "⚠️ Large raw prompt detected. This may bypass token reduction and burn tokens. "
                "Native Read/Grep on whole files pulls raw text into the model context. "
                "Prefer: keep the prompt short, pass paths as --inputs, then run /token-reducer (or the CLI) first. "
                f"(approx_tokens={estimate_tokens(prompt)})"
            )

    if turns >= critical_reset_turn and turns % 10 == 0:
        messages.append(
            f"🚨 CRITICAL: Session has reached {turns} turns. Context window is likely near capacity. "
            "Token efficiency is severely degraded. START A NEW CHAT NOW to restore full token savings. "
            "Run /compact first to preserve important context, then begin a fresh session."
        )
    elif turns >= auto_reset_turn and turns % 5 == 0:
        messages.append(
            f"🔄 AUTO-RESET RECOMMENDED: Session has reached {turns} turns. Chat history is consuming "
            "significant tokens and reducing the effectiveness of token-reducer. "
            "Strongly recommended: run /compact now, then start a fresh chat session."
        )
    elif turns == auto_compact_turn:
        messages.append(
            f"📦 AUTO-COMPACT SUGGESTED: You've reached {auto_compact_turn} turns. Run /compact now to compress "
            "conversation history and reclaim context window space. This keeps token-reducer effective."
        )
    elif turns in reminder_turns:
        messages.append(
            "🧹 Session hygiene reminder: context history is growing. Run /compact at milestones and start a fresh chat when switching major tasks."
        )

    if messages:
        result["systemMessage"] = "\n\n".join(messages)

    print(json.dumps(result) if result else json.dumps({}), file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
