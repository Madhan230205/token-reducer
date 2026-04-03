from __future__ import annotations

import re
from pathlib import Path

from .chunker import estimate_tokens, is_code_file, tokenize
from .config import get_weight
from .models import (
    Candidate,
    CandidateSummary,
    ContextPacket,
    RetrievalInfo,
    TokenMetrics,
)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def textrank_score_sentences(
    sentences: list[str], damping: float = 0.85, iterations: int = 30
) -> list[tuple[float, str]]:
    """Score sentences using TextRank algorithm for extractive summarization.

    TextRank builds a graph where sentences are nodes and edges are weighted
    by semantic similarity. Sentences that are similar to many other important
    sentences receive higher scores - capturing semantic centrality.
    """
    if len(sentences) <= 2:
        return [(1.0, s) for s in sentences]

    n = len(sentences)

    # Build similarity matrix using Jaccard similarity of word sets
    similarity_matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    sentence_words = [set(tokenize(s)) for s in sentences]

    for i in range(n):
        for j in range(i + 1, n):
            if not sentence_words[i] or not sentence_words[j]:
                sim = 0.0
            else:
                intersection = len(sentence_words[i] & sentence_words[j])
                union = len(sentence_words[i] | sentence_words[j])
                sim = intersection / union if union > 0 else 0.0
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim

    # Normalize rows (outgoing edge weights)
    for i in range(n):
        row_sum = sum(similarity_matrix[i])
        if row_sum > 0:
            for j in range(n):
                similarity_matrix[i][j] /= row_sum

    # Initialize scores uniformly
    scores = [1.0 / n] * n

    # Power iteration
    for _ in range(iterations):
        new_scores = [0.0] * n
        for i in range(n):
            rank_sum = sum(similarity_matrix[j][i] * scores[j] for j in range(n))
            new_scores[i] = (1 - damping) / n + damping * rank_sum
        scores = new_scores

    # Normalize final scores
    max_score = max(scores) if scores else 1.0
    if max_score > 0:
        scores = [s / max_score for s in scores]

    return list(zip(scores, sentences, strict=False))


def cluster_chunks_semantically(
    chunks: list[str],
    embeddings: list[list[float]],
    num_clusters: int = 5,
) -> list[list[int]]:
    """Cluster chunks by semantic similarity using k-means on embeddings.

    Returns list of clusters, each containing chunk indices.
    This enables selecting diverse, representative chunks.
    """
    if len(chunks) <= num_clusters:
        return [[i] for i in range(len(chunks))]

    try:
        import numpy as np
    except ImportError:
        # Fallback: return all chunks as single cluster
        return [list(range(len(chunks)))]

    embeddings_arr = np.array(embeddings)
    n = len(chunks)
    k = min(num_clusters, n)

    # Simple k-means clustering
    # Initialize centroids randomly
    rng = np.random.default_rng(42)
    centroid_indices = rng.choice(n, size=k, replace=False)
    centroids = embeddings_arr[centroid_indices].copy()

    assignments = np.zeros(n, dtype=int)

    for _ in range(20):  # Max iterations
        # Assign points to nearest centroid
        for i in range(n):
            distances = [np.linalg.norm(embeddings_arr[i] - centroids[j]) for j in range(k)]
            assignments[i] = int(np.argmin(distances))

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            cluster_points = embeddings_arr[assignments == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    # Build cluster lists
    clusters: list[list[int]] = [[] for _ in range(k)]
    for i, cluster_id in enumerate(assignments):
        clusters[cluster_id].append(i)

    return [c for c in clusters if c]  # Remove empty clusters


def trim_to_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


_CODE_SIGNATURE_RE = re.compile(
    r"^(?:"
    r"(?:export\s+)?(?:async\s+)?(?:def|function|fn|func|fun|sub)\s+\w+[^{;]*"
    r"|(?:public|private|protected|internal|static|abstract|final|override|open|sealed|suspend|inline)\s+.*\w+\s*\([^)]*\)"
    r"|class\s+\w+[^{]*"
    r"|interface\s+\w+[^{]*"
    r"|struct\s+\w+[^{]*"
    r"|enum\s+\w+[^{]*"
    r"|trait\s+\w+[^{]*"
    r"|impl\s+.*"
    r"|type\s+\w+\s+(?:struct|interface)[^{]*"
    r"|module\s+\w+"
    r"|(?:const|let|var)\s+\w+\s*[:=]"
    r")",
    re.MULTILINE,
)

_DOCSTRING_RE = re.compile(
    r'(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|/\*\*[\s\S]*?\*/|///.*|//!.*|##.*)',
)


def extract_code_signatures(text: str) -> list[str]:
    """Extract function/class signatures and docstrings from code text."""
    results: list[str] = []
    for match in _CODE_SIGNATURE_RE.finditer(text):
        sig = match.group(0).strip()
        if sig and len(sig.split()) >= 2:
            results.append(sig)
    for match in _DOCSTRING_RE.finditer(text):
        doc = match.group(0).strip()
        # Trim long docstrings to first 3 lines
        doc_lines = doc.splitlines()
        if len(doc_lines) > 3:
            doc = "\n".join(doc_lines[:3]) + " ..."
        if doc and len(doc.split()) >= 3:
            results.append(doc)
    return results


def merge_adjacent_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Merge candidates with adjacent chunk indices from the same source to eliminate overlap."""
    if len(candidates) <= 1:
        return candidates

    # Group by source
    by_source: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_source.setdefault(c.source, []).append(c)

    merged: list[Candidate] = []
    for _source, group in by_source.items():
        group.sort(key=lambda c: c.chunk_index)
        run: list[Candidate] = [group[0]]

        for i in range(1, len(group)):
            prev = run[-1]
            curr = group[i]
            if curr.chunk_index == prev.chunk_index + 1:
                run.append(curr)
            else:
                merged.append(_merge_run(run))
                run = [curr]
        merged.append(_merge_run(run))

    # Preserve original score ordering
    merged.sort(key=lambda c: c.final_score, reverse=True)
    return merged


def _merge_run(run: list[Candidate]) -> Candidate:
    if len(run) == 1:
        return run[0]
    combined_text = "\n".join(c.text for c in run)
    return Candidate(
        chunk_id=run[0].chunk_id,
        source=run[0].source,
        chunk_index=run[0].chunk_index,
        text=combined_text,
        token_estimate=sum(c.token_estimate for c in run),
        bm25_score=run[0].bm25_score,
        fts_rank=run[0].fts_rank,
        vector_rank=run[0].vector_rank,
        vector_score=max(c.vector_score for c in run),
        fts_score=max(c.fts_score for c in run),
        overlap_score=max(c.overlap_score for c in run),
        final_score=max(c.final_score for c in run),
    )


def compress_candidates(
    query: str,
    candidates: list[Candidate],
    word_budget: int,
    relevance_floor: float = 0.15,
) -> list[str]:
    """Compress candidate chunks into citation-rich summary bullets.

    Args:
        query: The user query for relevance scoring.
        candidates: Ranked candidates from retrieval.
        word_budget: Maximum words in compressed output.
        relevance_floor: Minimum final_score threshold. Chunks below this
            score are rejected to preserve context purity and API costs.
    """
    # Merge adjacent chunks before compression to eliminate overlap redundancy
    candidates = merge_adjacent_candidates(candidates)

    query_terms = set(tokenize(query))
    bullets: list[str] = []
    words_used = 0

    # Get configurable weights for sentence scoring
    length_bonus_w = get_weight("sentence_length_bonus_weight")
    length_normalizer = get_weight("sentence_length_normalizer")

    for candidate in candidates:
        # Knapsack Relevance Floor: if the chunk's score is too low, STOP packing.
        # This prevents pulling in noise just to hit a budget.
        if candidate.final_score < relevance_floor:
            break
        source_is_code = is_code_file(candidate.source)

        if source_is_code:
            # For code: extract signatures and docstrings instead of sentence splitting
            snippets = extract_code_signatures(candidate.text)
            if not snippets:
                # Fallback: use first N lines preserving code structure
                code_lines = [line for line in candidate.text.splitlines() if line.strip()]
                snippets = code_lines[:5] if code_lines else []
            if not snippets:
                continue
            summary = " | ".join(snippets[:4]).strip()
        else:
            # For prose: TextRank-based sentence scoring for intelligent summarization
            sentences = split_sentences(candidate.text)
            if not sentences:
                continue

            # Get configurable weights
            textrank_w = get_weight("textrank_weight")
            query_relevance_w = get_weight("query_relevance_weight")

            # Use TextRank for semantic importance, boosted by query overlap
            textrank_scores = textrank_score_sentences(sentences)
            scored_sentences: list[tuple[float, str]] = []

            for tr_score, sentence in textrank_scores:
                sent_terms = set(tokenize(sentence))
                overlap = len(query_terms & sent_terms)
                query_signal = overlap / float(len(query_terms)) if query_terms else 0.0
                length_bonus = min(1.0, len(sentence.split()) / length_normalizer)
                # Combine TextRank centrality with query relevance
                combined_score = (
                    (textrank_w * tr_score)
                    + (query_relevance_w * query_signal)
                    + (length_bonus_w * length_bonus)
                )
                scored_sentences.append((combined_score, sentence))

            scored_sentences.sort(key=lambda pair: pair[0], reverse=True)
            selected = [s for _, s in scored_sentences[:2]]
            summary = " ".join(selected).strip()

        if not summary:
            continue

        citation = f"[{Path(candidate.source).name}#chunk-{candidate.chunk_index}]"
        bullet = f"{summary} {citation}".strip()
        bullet_words = len(bullet.split())

        if bullets and words_used + bullet_words > word_budget:
            continue
        if not bullets and bullet_words > word_budget:
            bullet = trim_to_words(bullet, max_words=word_budget)
            bullet_words = len(bullet.split())

        bullets.append(bullet)
        words_used += bullet_words
        if words_used >= word_budget:
            break

    if bullets:
        return bullets

    fallback: list[str] = []
    for candidate in candidates[:3]:
        snippet = " ".join(candidate.text.split()[:35]).strip()
        if snippet:
            fallback.append(
                f"{snippet}... [{Path(candidate.source).name}#chunk-{candidate.chunk_index}]"
            )
    return fallback


def build_packet(
    query: str,
    selected: list[Candidate],
    candidate_pool: list[Candidate],
    bullets: list[str],
    fts_hit_count: int,
    vector_hit_count: int,
    hybrid_mode: str,
    retrieval_mode: str,
    vector_backend_used: str,
    vector_model_used: str | None,
    vector_retrieval_path: str,
    referenced_symbols: list[dict] | None = None,
    imported_context: list[Candidate] | None = None,
) -> ContextPacket:
    source_count = len({c.source for c in selected})
    candidate_pool_tokens = sum(c.token_estimate for c in candidate_pool)
    selected_token_estimate = sum(c.token_estimate for c in selected)
    compressed_token_estimate = sum(estimate_tokens(b) for b in bullets)
    estimated_savings = max(0, selected_token_estimate - compressed_token_estimate)
    savings_from_pool = max(0, candidate_pool_tokens - compressed_token_estimate)
    savings_pct_pool = (
        round((savings_from_pool / candidate_pool_tokens) * 100.0, 2)
        if candidate_pool_tokens > 0
        else 0.0
    )

    lines: list[str] = [
        "CONTEXT_PACKET_START",
        f"query: {query}",
        f"selected_chunks: {len(selected)}",
        f"sources: {source_count}",
        f"retrieval_mode: {retrieval_mode}",
        f"hybrid_mode: {hybrid_mode}",
        "fts_ranking: bm25",
        f"vector_backend_used: {vector_backend_used}",
        f"vector_retrieval_path: {vector_retrieval_path}",
    ]
    if vector_model_used:
        lines.append(f"vector_model_used: {vector_model_used}")
    lines.extend(
        [
            f"fts_hits: {fts_hit_count}",
            f"vector_hits: {vector_hit_count}",
            f"estimated_token_savings: {estimated_savings}",
            f"candidate_pool_token_reduction_pct: {savings_pct_pool}",
            "",
            "compressed_context:",
        ]
    )
    for bullet in bullets:
        lines.append(f"- {bullet}")
    lines.extend(["", "CONTEXT_PACKET_END"])

    retrieval = RetrievalInfo(
        mode=retrieval_mode,
        hybrid_mode=hybrid_mode,
        bm25_enabled=True,
        fts_hits=fts_hit_count,
        vector_hits=vector_hit_count,
        vector_backend_used=vector_backend_used,
        vector_model_used=vector_model_used,
        vector_retrieval_path=vector_retrieval_path,
        candidate_pool_size=len(candidate_pool),
    )

    token_metrics = TokenMetrics(
        candidate_pool_tokens=candidate_pool_tokens,
        selected_chunk_tokens=selected_token_estimate,
        compressed_tokens=compressed_token_estimate,
        payload_tokens=compressed_token_estimate,
        savings_from_selected_tokens=estimated_savings,
        savings_from_candidate_pool_tokens=savings_from_pool,
        savings_from_candidate_pool_pct=savings_pct_pool,
    )

    candidates = [
        CandidateSummary(
            chunk_id=c.chunk_id,
            source=c.source,
            chunk_index=c.chunk_index,
            bm25_score=round(float(c.bm25_score), 6) if c.bm25_score is not None else None,
            fts_score=round(c.fts_score, 6),
            vector_score=round(c.vector_score, 6),
            overlap_score=round(c.overlap_score, 6),
            final_score=round(c.final_score, 6),
            fts_rank=c.fts_rank,
            vector_rank=c.vector_rank,
        )
        for c in selected
    ]

    return ContextPacket(
        query=query,
        selected_chunks=len(selected),
        source_count=source_count,
        estimated_token_savings=estimated_savings,
        retrieval=retrieval,
        token_metrics=token_metrics,
        bullets=bullets,
        candidates=candidates,
        packet="\n".join(lines),
    )
