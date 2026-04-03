from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from hashlib import blake2b

from .chunker import char_ngrams, tokenize
from .config import _EMBEDDING_MODEL_CACHE, get_weight


def embed_text_hash(text: str, dimensions: int) -> list[float]:
    """Create hash-based embeddings using locality-sensitive hashing.

    Note: Hash embeddings provide lexical similarity similar to FTS5/BM25.
    When hashEmbeddingSkipVector is true (default), vector retrieval is
    skipped entirely when using hash backend to avoid redundant noise.
    """
    vec = [0.0] * dimensions
    terms = tokenize(text)
    if not terms:
        return vec

    ngram_weight = get_weight("char_ngram_weight")

    term_counts: dict[str, float] = {}
    for term in terms:
        term_counts[f"w:{term}"] = term_counts.get(f"w:{term}", 0.0) + 1.0
        for gram in char_ngrams(term):
            key = f"g:{gram}"
            term_counts[key] = term_counts.get(key, 0.0) + ngram_weight

    for term, count in term_counts.items():
        digest = blake2b(term.encode("utf-8"), digest_size=16).digest()
        idx = int.from_bytes(digest[:8], byteorder="big") % dimensions
        sign = 1.0 if digest[8] % 2 == 0 else -1.0
        weight = 1.0 + math.log1p(count)
        vec[idx] += sign * weight

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def get_sentence_transformer_model(model_name: str):
    cached = _EMBEDDING_MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "ML embedding backend requested but sentence-transformers is not installed."
        ) from exc

    model = SentenceTransformer(model_name)
    _EMBEDDING_MODEL_CACHE[model_name] = model
    return model


def embed_text_ml(text: str, model_name: str) -> list[float]:
    model = get_sentence_transformer_model(model_name)
    embedding = model.encode([text], normalize_embeddings=True)[0]
    return [float(x) for x in embedding]


def resolve_embedding_backend(
    requested_backend: str,
    requested_model: str,
) -> tuple[str, str | None]:
    backend = requested_backend.strip().lower()
    if backend == "hash":
        return "hash", None

    if backend == "ml":
        try:
            get_sentence_transformer_model(requested_model)
            return "ml", requested_model
        except Exception as exc:
            print(
                f"[warn] ML embedding backend unavailable ({exc}). Falling back to hash embeddings.",
                file=sys.stderr,
            )
            return "hash", None

    print(
        f"[warn] Unknown embedding backend '{requested_backend}'. Falling back to hash.",
        file=sys.stderr,
    )
    return "hash", None


def embed_text(
    text: str,
    dimensions: int,
    embedding_backend: str,
    embedding_model: str | None,
) -> tuple[list[float], str, str | None]:
    backend = embedding_backend.strip().lower()
    if backend == "ml" and embedding_model:
        try:
            return embed_text_ml(text=text, model_name=embedding_model), "ml", embedding_model
        except Exception as exc:
            print(
                f"[warn] ML embedding runtime failed ({exc}). Falling back to hash embeddings.",
                file=sys.stderr,
            )

    return embed_text_hash(text=text, dimensions=dimensions), "hash", None


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return float(sum(a * b for a, b in zip(vec_a, vec_b, strict=False)))
