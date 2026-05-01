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
    DEFAULT_RELEVANCE_FLOOR,
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

    return TokenReducerRuntimeConfig(
        chunk_size_words=chunk,
        chunk_overlap_words=overlap,
        compression_word_budget=budget,
        default_top_k=top_k,
        relevance_floor=floor,
    )

