from __future__ import annotations

import math

from token_reducer.embeddings import (
    cosine_similarity,
    embed_text,
    embed_text_hash,
    resolve_embedding_backend,
)

# ---------------------------------------------------------------------------
# embed_text_hash
# ---------------------------------------------------------------------------


class TestEmbedTextHash:
    def test_returns_correct_length(self) -> None:
        vec = embed_text_hash("hello world", dimensions=128)
        assert len(vec) == 128

    def test_custom_dimensions(self) -> None:
        assert len(embed_text_hash("text", dimensions=32)) == 32
        assert len(embed_text_hash("text", dimensions=512)) == 512

    def test_normalized_to_unit_length(self) -> None:
        vec = embed_text_hash("the quick brown fox", dimensions=64)
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-6

    def test_empty_text_returns_zero_vector(self) -> None:
        vec = embed_text_hash("", dimensions=64)
        assert vec == [0.0] * 64

    def test_deterministic(self) -> None:
        assert embed_text_hash("same text", 64) == embed_text_hash("same text", 64)

    def test_different_texts_produce_different_vectors(self) -> None:
        vec_a = embed_text_hash("hello world", dimensions=64)
        vec_b = embed_text_hash("completely different xyz", dimensions=64)
        assert vec_a != vec_b

    def test_similar_texts_score_higher_than_unrelated(self) -> None:
        # Texts sharing tokens should have higher cosine similarity
        vec_a = embed_text_hash("machine learning models training", dimensions=256)
        vec_b = embed_text_hash("machine learning algorithms training", dimensions=256)
        vec_c = embed_text_hash("gardening vegetables tomatoes soil", dimensions=256)
        sim_ab = cosine_similarity(vec_a, vec_b)
        sim_ac = cosine_similarity(vec_a, vec_c)
        assert sim_ab > sim_ac


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_self_similarity_is_one(self) -> None:
        vec = embed_text_hash("test vector", dimensions=64)
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_empty_vectors_return_zero(self) -> None:
        assert cosine_similarity([], []) == 0.0

    def test_one_empty_vector_returns_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0], []) == 0.0

    def test_length_mismatch_returns_zero(self) -> None:
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0

    def test_orthogonal_vectors_are_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == 0.0

    def test_identical_unit_vectors_are_one(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_opposite_vectors_are_minus_one(self) -> None:
        assert abs(cosine_similarity([1.0, 0.0], [-1.0, 0.0]) - (-1.0)) < 1e-6

    def test_symmetric(self) -> None:
        a = [0.6, 0.8]
        b = [0.8, 0.6]
        assert abs(cosine_similarity(a, b) - cosine_similarity(b, a)) < 1e-10


# ---------------------------------------------------------------------------
# resolve_embedding_backend
# ---------------------------------------------------------------------------


class TestResolveEmbeddingBackend:
    def test_hash_returns_hash_and_none_model(self) -> None:
        backend, model = resolve_embedding_backend("hash", "any-model")
        assert backend == "hash"
        assert model is None

    def test_hash_is_case_insensitive(self) -> None:
        backend, _ = resolve_embedding_backend("HASH", "model")
        assert backend == "hash"

    def test_hash_with_whitespace(self) -> None:
        backend, _ = resolve_embedding_backend("  hash  ", "model")
        assert backend == "hash"

    def test_unknown_backend_falls_back_to_hash(self) -> None:
        backend, model = resolve_embedding_backend("bogus_backend", "some-model")
        assert backend == "hash"
        assert model is None

    def test_ml_without_sentence_transformers_falls_back(self) -> None:
        # If sentence-transformers is not installed, ml falls back to hash
        backend, model = resolve_embedding_backend("ml", "nonexistent-model-abc-xyz-12345")
        assert backend in ("ml", "hash")  # graceful degradation


# ---------------------------------------------------------------------------
# embed_text (dispatcher)
# ---------------------------------------------------------------------------


class TestEmbedText:
    def test_hash_backend_returns_correct_shape(self) -> None:
        vec, backend, model = embed_text("hello world", 64, "hash", None)
        assert len(vec) == 64
        assert backend == "hash"
        assert model is None

    def test_hash_backend_vector_is_normalized(self) -> None:
        vec, _, _ = embed_text("the quick brown fox", 64, "hash", None)
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-6

    def test_ml_with_no_model_falls_back_to_hash(self) -> None:
        # embedding_model=None → ml condition is False → hash path
        vec, backend, model = embed_text("test content", 64, "ml", None)
        assert len(vec) == 64
        assert backend == "hash"
        assert model is None

    def test_unknown_backend_falls_back_to_hash(self) -> None:
        vec, backend, model = embed_text("text", 64, "unknown_backend", None)
        assert backend == "hash"
        assert model is None

    def test_deterministic_across_calls(self) -> None:
        vec1, _, _ = embed_text("same text", 128, "hash", None)
        vec2, _, _ = embed_text("same text", 128, "hash", None)
        assert vec1 == vec2
