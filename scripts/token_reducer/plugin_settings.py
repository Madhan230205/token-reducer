"""Load token-reducer defaults from plugin settings.json (CLAUDE_PLUGIN_ROOT or repo root)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LSP_SERVERS,
    DEFAULT_RELEVANCE_FLOOR,
    DEFAULT_SHADOW_LINTER_CMDS,
    DEFAULT_SHADOW_LINTER_TIMEOUT,
    DEFAULT_SHADOW_LINTER_TIMEOUT_BY_EXT,
    DEFAULT_TOP_K,
    DEFAULT_WORD_BUDGET,
)


def _candidate_settings_paths() -> list[Path]:
    paths: list[Path] = []
    root = os.environ.get("CLAUDE_PLUGIN_ROOT", "").strip()
    if root:
        paths.append(Path(root) / "settings.json")
    # Dev checkout: scripts/token_reducer/plugin_settings.py -> repo root
    here = Path(__file__).resolve()
    if here.parent.name == "token_reducer" and here.parent.parent.name == "scripts":
        paths.append(here.parent.parent.parent / "settings.json")
    return paths


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _first_settings_blob() -> dict[str, Any]:
    for path in _candidate_settings_paths():
        data = _read_json(path)
        if data and "tokenReducer" in data and isinstance(data["tokenReducer"], dict):
            return dict(data["tokenReducer"])
    return {}


@dataclass(frozen=True)
class TokenReducerRuntimeConfig:
    chunk_size_words: int
    chunk_overlap_words: int
    compression_word_budget: int
    default_top_k: int
    relevance_floor: float
    shadow_linter_cmds: dict[str, str]
    shadow_linter_timeout: int
    shadow_linter_timeout_by_ext: dict[str, int]
    lsp_servers: dict[str, list[str]]


@lru_cache(maxsize=1)
def get_runtime_defaults() -> TokenReducerRuntimeConfig:
    raw = _first_settings_blob()

    def _int(key: str, default: int) -> int:
        v = raw.get(key)
        if isinstance(v, bool) or v is None:
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _float(key: str, default: float) -> float:
        v = raw.get(key)
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    chunk = _int("chunkSizeWords", DEFAULT_CHUNK_SIZE)
    overlap = _int("chunkOverlapWords", DEFAULT_CHUNK_OVERLAP)
    budget = _int("compressionWordBudget", DEFAULT_WORD_BUDGET)
    top_k = _int("defaultTopK", _int("maxFinalContexts", DEFAULT_TOP_K))
    if top_k < 1:
        top_k = DEFAULT_TOP_K
    floor = _float("relevanceFloor", float(DEFAULT_RELEVANCE_FLOOR))

    shadow_cmds = {str(k).lower(): str(v) for k, v in DEFAULT_SHADOW_LINTER_CMDS.items()}
    raw_shadow = raw.get("shadowLinterCmds")
    if isinstance(raw_shadow, dict):
        for k, v in raw_shadow.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            key = k if k.startswith(".") else f".{k}"
            shadow_cmds[key.lower()] = v

    shadow_timeout = _int("shadowLinterTimeout", DEFAULT_SHADOW_LINTER_TIMEOUT)
    if shadow_timeout < 1:
        shadow_timeout = DEFAULT_SHADOW_LINTER_TIMEOUT

    shadow_timeouts = {
        str(k).lower(): int(v) for k, v in DEFAULT_SHADOW_LINTER_TIMEOUT_BY_EXT.items()
    }
    raw_timeouts = raw.get("shadowLinterTimeoutByExt")
    if isinstance(raw_timeouts, dict):
        for k, v in raw_timeouts.items():
            if not isinstance(k, str):
                continue
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if iv < 1:
                continue
            key = k if k.startswith(".") else f".{k}"
            shadow_timeouts[key.lower()] = iv

    lsp_servers: dict[str, list[str]] = {k.lower(): list(v) for k, v in DEFAULT_LSP_SERVERS.items()}
    raw_lsp = raw.get("lspServers")
    if isinstance(raw_lsp, dict):
        for k, v in raw_lsp.items():
            if not isinstance(k, str) or not isinstance(v, list):
                continue
            if not all(isinstance(x, str) for x in v):
                continue
            if not v:
                continue
            key = k if k.startswith(".") else f".{k}"
            lsp_servers[key.lower()] = list(v)

    return TokenReducerRuntimeConfig(
        chunk_size_words=chunk,
        chunk_overlap_words=overlap,
        compression_word_budget=budget,
        default_top_k=top_k,
        relevance_floor=floor,
        shadow_linter_cmds=shadow_cmds,
        shadow_linter_timeout=shadow_timeout,
        shadow_linter_timeout_by_ext=shadow_timeouts,
        lsp_servers=lsp_servers,
    )
