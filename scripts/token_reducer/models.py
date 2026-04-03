from __future__ import annotations

import re
from datetime import UTC, datetime
from hashlib import blake2b
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Candidate(BaseModel):
    """A chunk candidate for retrieval ranking."""

    model_config = ConfigDict(frozen=False, validate_assignment=True)

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


class Chunk(BaseModel):
    """A text chunk from a document."""

    model_config = ConfigDict(frozen=True)

    document_id: int
    chunk_index: int
    text: str
    token_estimate: int


class ChunkEmbedding(BaseModel):
    """Embedding data for a chunk."""

    model_config = ConfigDict(frozen=True)

    chunk_id: int
    embedding: list[float]
    backend: str = "hash"
    dimensions: int = 256
    model_name: str | None = None


class Document(BaseModel):
    """A document in the index."""

    model_config = ConfigDict(frozen=True)

    id: int | None = None
    source: str
    raw_text: str
    cleaned_text: str
    created_at: str
    updated_at: str


class QueryEmbedding(BaseModel):
    """Cached query embedding."""

    model_config = ConfigDict(frozen=True)

    query_key: str
    query_text_hash: str
    embedding: list[float]
    backend: str
    dimensions: int
    model_name: str | None = None


class TokenMetrics(BaseModel):
    """Token usage metrics for a context packet."""

    model_config = ConfigDict(frozen=True)

    candidate_pool_tokens: int
    selected_chunk_tokens: int
    compressed_tokens: int
    payload_tokens: int
    savings_from_selected_tokens: int
    savings_from_candidate_pool_tokens: int
    savings_from_candidate_pool_pct: float


class RetrievalInfo(BaseModel):
    """Retrieval metadata for a context packet."""

    model_config = ConfigDict(frozen=True)

    mode: str
    hybrid_mode: str
    bm25_enabled: bool = True
    fts_hits: int
    vector_hits: int
    vector_backend_used: str
    vector_model_used: str | None
    vector_retrieval_path: str
    candidate_pool_size: int


class CandidateSummary(BaseModel):
    """Summary of a candidate for packet output."""

    model_config = ConfigDict(frozen=True)

    chunk_id: int
    source: str
    chunk_index: int
    bm25_score: float | None = None
    fts_score: float
    vector_score: float
    overlap_score: float
    final_score: float
    fts_rank: int | None = None
    vector_rank: int | None = None


class SessionMemory(BaseModel):
    """Session memory for a context packet."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    recent_queries: list[str] = Field(default_factory=list)


class CacheInfo(BaseModel):
    """Cache information for a context packet."""

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    hit: bool = False
    key: str = ""
    ttl_seconds: int = 0


class ContextPacket(BaseModel):
    """Complete context packet returned by the pipeline."""

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    query: str
    selected_chunks: int
    source_count: int
    estimated_token_savings: int
    retrieval: RetrievalInfo
    token_metrics: TokenMetrics
    bullets: list[str]
    candidates: list[CandidateSummary]
    packet: str
    session_memory: SessionMemory | None = None
    cache: CacheInfo | None = None


class Symbol(BaseModel):
    """A code symbol extracted from a chunk."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: str
    signature: str | None = None


class FileDependency(BaseModel):
    """A file dependency relationship."""

    model_config = ConfigDict(frozen=True)

    source_file: str
    target_import: str
    resolved_target: str | None = None


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def utc_now_epoch() -> int:
    return int(datetime.now(UTC).timestamp())


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
