"""Lightweight local logging of pipeline outcomes (no ML, no training)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _default_log_path() -> Path:
    root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if root:
        return Path(root) / ".cache" / "token-reducer" / "feedback.jsonl"
    return Path.home() / ".cache" / "token-reducer" / "feedback.jsonl"


def log_result(
    prompt: str,
    context: str,
    response: str | None = None,
    *,
    extra: dict[str, Any] | None = None,
    log_path: Path | None = None,
) -> None:
    """Append one JSON line: context size, chunk hints, optional quality signal."""
    path = log_path or _default_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": int(time.time()),
            "prompt_chars": len(prompt or ""),
            "context_chars": len(context or ""),
            "context_words": len((context or "").split()),
            "response_chars": len(response or "") if response is not None else None,
            "extra": extra or {},
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        return


def read_feedback_source_boosts(
    log_path: Path | None = None,
    *,
    tail_lines: int = 72,
    min_words: int = 24,
    max_boost: float = 0.06,
) -> dict[str, float]:
    """Lightweight ranking hints from recent JSONL: boost sources seen in 'fat' outcomes.

    Reads the last ``tail_lines`` of the log; rows with ``context_words`` above ``min_words``
    contribute ``extra.selected_sources`` path basenames with a capped positive delta.
    """
    path = log_path or _default_log_path()
    if not path.is_file():
        return {}
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    rows = raw[-tail_lines:]
    counts: dict[str, int] = {}
    for line in rows:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        words = int(obj.get("context_words") or 0)
        if words < min_words:
            continue
        extra = obj.get("extra")
        if not isinstance(extra, dict):
            continue
        srcs = extra.get("selected_sources")
        if not isinstance(srcs, list):
            continue
        for s in srcs:
            if not isinstance(s, str) or not s.strip():
                continue
            key = Path(s).name
            counts[key] = counts.get(key, 0) + 1
    out: dict[str, float] = {}
    for name, n in counts.items():
        out[name] = min(max_boost, 0.012 * (n**0.5))
    return out


def _strategy_prune_deltas_from_jsonl(
    path: Path,
    *,
    tail_lines: int = 200,
    min_samples: int = 4,
    fat_words: int = 200,
    lean_words: int = 72,
) -> dict[str, int]:
    if not path.is_file():
        return {}
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    rows = raw[-tail_lines:]
    buckets: dict[str, list[int]] = {}
    for line in rows:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        extra = obj.get("extra")
        if not isinstance(extra, dict):
            continue
        sid = extra.get("strategy_id")
        if not isinstance(sid, str) or not sid.strip():
            continue
        words = int(obj.get("context_words") or 0)
        buckets.setdefault(sid.strip(), []).append(words)
    out: dict[str, int] = {}
    for sid, ws in buckets.items():
        if len(ws) < min_samples:
            continue
        med = sorted(ws)[len(ws) // 2]
        if med >= fat_words:
            out[sid] = -1
        elif med <= lean_words:
            out[sid] = 1
        else:
            out[sid] = 0
    return out


def strategy_prune_deltas_from_feedback_log(
    log_path: Path | None = None,
    *,
    tail_lines: int = 200,
    min_samples: int = 4,
    fat_words: int = 200,
    lean_words: int = 72,
) -> dict[str, int]:
    """Snapshot prune deltas from the feedback JSONL only (for persistence hooks)."""
    path = log_path or _default_log_path()
    return _strategy_prune_deltas_from_jsonl(
        path,
        tail_lines=tail_lines,
        min_samples=min_samples,
        fat_words=fat_words,
        lean_words=lean_words,
    )


def _workspace_prune_ema_path(workspace_root: Path) -> Path:
    return workspace_root / ".token-reducer" / "prune_ema.json"


def load_workspace_prune_ema(workspace_root: Path | None) -> dict[str, float]:
    if workspace_root is None:
        return {}
    path = _workspace_prune_ema_path(workspace_root)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    ema = data.get("ema")
    if not isinstance(ema, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in ema.items():
        if isinstance(k, str) and isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def persist_workspace_prune_ema(workspace_root: Path | None, log_deltas: dict[str, int]) -> None:
    """Smooth logged prune signals into a per-workspace file (cross-session)."""
    if workspace_root is None or not log_deltas:
        return
    path = _workspace_prune_ema_path(workspace_root)
    prev = load_workspace_prune_ema(workspace_root)
    ema: dict[str, float] = dict(prev)
    for sid, d in log_deltas.items():
        ema[sid] = 0.82 * ema.get(sid, 0.0) + 0.18 * float(int(d))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ema": ema}, indent=2), encoding="utf-8")
    except OSError:
        return


def read_strategy_prune_adjustments(
    log_path: Path | None = None,
    *,
    tail_lines: int = 200,
    min_samples: int = 4,
    fat_words: int = 200,
    lean_words: int = 72,
    workspace_root: Path | None = None,
) -> dict[str, int]:
    """From feedback JSONL plus optional workspace EMA, suggest small ``prune_k`` deltas per strategy."""
    path = log_path or _default_log_path()
    log_d = _strategy_prune_deltas_from_jsonl(
        path,
        tail_lines=tail_lines,
        min_samples=min_samples,
        fat_words=fat_words,
        lean_words=lean_words,
    )
    if workspace_root is None:
        return log_d
    persisted = load_workspace_prune_ema(workspace_root)
    keys = set(log_d) | set(persisted)
    merged: dict[str, int] = {}
    for sid in keys:
        v = 0.55 * float(log_d.get(sid, 0)) + 0.45 * float(persisted.get(sid, 0.0))
        merged[sid] = max(-1, min(1, int(round(v))))
    return merged
