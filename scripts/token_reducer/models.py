from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import blake2b
from pathlib import Path


@dataclass
class Candidate:
    chunk_id: int
    source: str
    chunk_index: int
    text: str
    token_estimate: int
    bm25_score: float | None = None
    fts_rank: int | None = None
    vector_rank: int | None = None
    fts_score: float = 0.0
    vector_score: float = 0.0
    overlap_score: float = 0.0
    final_score: float = 0.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return cleaned[:120] if cleaned else "default"


def hash_text(value: str) -> str:
    return blake2b(value.encode("utf-8"), digest_size=16).hexdigest()


def embedding_cache_key(
    text: str,
    embedding_backend: str,
    embedding_model: str | None,
    dimensions: int,
) -> str:
    return "|".join(
        [
            embedding_backend,
            embedding_model or "",
            str(dimensions),
            hash_text(text),
        ]
    )


def ann_artifact_prefix(
    db_path: Path,
    backend: str,
    dimensions: int,
    model_name: str | None,
) -> Path:
    model_slug = safe_slug(model_name or "none")
    stem = f"{safe_slug(backend)}_{dimensions}_{model_slug}"
    return db_path.parent / "ann" / stem
