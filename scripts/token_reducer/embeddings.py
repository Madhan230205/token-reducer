from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from hashlib import blake2b
from pathlib import Path

from .chunker import char_ngrams, tokenize
from .config import (
    _EMBEDDING_MODEL_CACHE,
    _ONNX_SESSION_CACHE,
    DEFAULT_ONNX_MAX_LENGTH,
    get_weight,
)


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


# Candidate ONNX filenames tried in order: quantized int8 first (smallest/fastest),
# then subfolder full-precision, then legacy root-level model.onnx.
_ONNX_CANDIDATE_FILENAMES = ["onnx/model_quantized.onnx", "onnx/model.onnx", "model.onnx"]


def get_onnx_session(model_path: str):
    """Load or retrieve cached ONNX Runtime session for embeddings."""
    cached = _ONNX_SESSION_CACHE.get(model_path)
    if cached is not None:
        return cached

    try:
        import onnxruntime as ort  # type: ignore
        from tokenizers import Tokenizer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "ONNX embedding backend requested but onnxruntime or tokenizers is not installed."
        ) from exc

    onnx_path: str | None = None
    tokenizer_path: str | None = None

    try:
        from huggingface_hub import hf_hub_download  # type: ignore

        for candidate in _ONNX_CANDIDATE_FILENAMES:
            try:
                onnx_path = hf_hub_download(repo_id=model_path, filename=candidate)
                break
            except Exception:
                continue

        if onnx_path is None:
            raise RuntimeError(f"No ONNX model file found in HuggingFace repo '{model_path}'")

        tokenizer_path = hf_hub_download(repo_id=model_path, filename="tokenizer.json")

    except Exception as hf_err:
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise RuntimeError(
                f"ONNX model not found at '{model_path}'. "
                "Provide a valid HuggingFace model ID or local directory path."
            ) from hf_err

        for candidate in _ONNX_CANDIDATE_FILENAMES:
            p = model_dir / candidate
            if p.exists():
                onnx_path = str(p)
                break

        if onnx_path is None:
            raise RuntimeError(f"No ONNX model file found locally in '{model_path}'") from hf_err

        tokenizer_path = str(model_dir / "tokenizer.json")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(tokenizer_path)

    _ONNX_SESSION_CACHE[model_path] = (session, tokenizer)
    return session, tokenizer


def embed_text_onnx(
    text: str, model_path: str, max_length: int = DEFAULT_ONNX_MAX_LENGTH
) -> list[float]:
    """Generate embeddings using ONNX Runtime for fast CPU inference.

    Uses attention-mask-weighted mean pooling so padding tokens are excluded
    from the average. Supports models that require token_type_ids.
    """
    import numpy as np

    session, tokenizer = get_onnx_session(model_path)

    encoding = tokenizer.encode(text)
    real_len = min(len(encoding.ids), max_length)
    pad_len = max_length - real_len

    tokens = encoding.ids[:real_len] + [0] * pad_len
    attn_mask = [1] * real_len + [0] * pad_len
    type_ids = (
        list(encoding.type_ids[:real_len]) + [0] * pad_len
        if encoding.type_ids
        else [0] * max_length
    )

    input_ids = np.array([tokens], dtype=np.int64)
    attention_mask_array = np.array([attn_mask], dtype=np.int64)
    token_type_ids_array = np.array([type_ids], dtype=np.int64)

    input_names = {inp.name for inp in session.get_inputs()}
    ort_inputs: dict[str, object] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask_array,
    }
    if "token_type_ids" in input_names:
        ort_inputs["token_type_ids"] = token_type_ids_array

    outputs = session.run(None, ort_inputs)

    # Attention-mask-weighted mean pooling: exclude padding tokens from average
    hidden = outputs[0][0]  # shape: (max_length, hidden_dim)
    mask = np.array(attn_mask, dtype=np.float32)
    masked_sum = (hidden * mask[:, np.newaxis]).sum(axis=0)
    embedding = masked_sum / (mask.sum() + 1e-9)

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return [float(x) for x in embedding]


def resolve_embedding_backend(
    requested_backend: str,
    requested_model: str,
) -> tuple[str, str | None]:
    backend = requested_backend.strip().lower()
    if backend == "hash":
        return "hash", None

    if backend == "onnx":
        try:
            get_onnx_session(requested_model)
            return "onnx", requested_model
        except Exception as exc:
            print(
                f"[warn] ONNX embedding backend unavailable ({exc}). Falling back to hash embeddings.",
                file=sys.stderr,
            )
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

    if backend == "onnx" and embedding_model:
        try:
            return embed_text_onnx(text=text, model_path=embedding_model), "onnx", embedding_model
        except Exception as exc:
            print(
                f"[warn] ONNX embedding runtime failed ({exc}). Falling back to hash embeddings.",
                file=sys.stderr,
            )

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
