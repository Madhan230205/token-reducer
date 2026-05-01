#!/usr/bin/env python3
"""
Comprehensive Benchmarking Suite for Token Reducer.

This module provides scientifically rigorous benchmarks that measure:
1. Token Savings - Before/after token counts with precise measurement
2. Latency Impact - p50/p95/p99 query latencies
3. Accuracy/Quality - Precision@K, Recall@K, MRR, NDCG metrics
4. Cost Analysis - Real dollar savings projections

Usage:
    python -m token_reducer.benchmark --corpus ./src --output results.json
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from .chunker import estimate_tokens
from .compressor import build_packet, compress_candidates
from .config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DIMENSIONS,
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FTS_K,
    DEFAULT_HYBRID_MODE,
    DEFAULT_MIN_FTS_HITS,
    DEFAULT_RELEVANCE_FLOOR,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_K,
    DEFAULT_WORD_BUDGET,
)
from .db import connect_db, index_corpus
from .embeddings import resolve_embedding_backend
from .pipeline import run_retrieval_pipeline


# ---------------------------------------------------------------------------
# Ground Truth Dataset for Accuracy Measurement
# ---------------------------------------------------------------------------

@dataclass
class RelevanceJudgment:
    """A query with known relevant chunks for accuracy measurement."""
    query: str
    relevant_keywords: list[str]  # Keywords that MUST appear in relevant results
    irrelevant_keywords: list[str] = field(default_factory=list)  # Keywords that indicate irrelevance
    expected_files: list[str] = field(default_factory=list)  # Expected source files


# Curated test queries with ground truth for accuracy measurement
GROUND_TRUTH_QUERIES: list[RelevanceJudgment] = [
    RelevanceJudgment(
        query="How does the retrieval pipeline work?",
        relevant_keywords=["retrieve", "fts", "vector", "bm25", "rerank", "candidate"],
        irrelevant_keywords=["test", "mock"],
        expected_files=["retriever.py", "pipeline.py"],
    ),
    RelevanceJudgment(
        query="Database schema and connection handling",
        relevant_keywords=["sqlite", "conn", "table", "index", "schema", "CREATE"],
        expected_files=["db.py"],
    ),
    RelevanceJudgment(
        query="Text compression and summarization",
        relevant_keywords=["compress", "textrank", "sentence", "bullet", "summary", "word_budget"],
        expected_files=["compressor.py"],
    ),
    RelevanceJudgment(
        query="Embedding generation and vector similarity",
        relevant_keywords=["embed", "vector", "cosine", "similarity", "dimension"],
        expected_files=["embeddings.py"],
    ),
    RelevanceJudgment(
        query="CLI argument parsing and commands",
        relevant_keywords=["typer", "option", "command", "argument", "cli"],
        expected_files=["cli.py"],
    ),
    RelevanceJudgment(
        query="Chunking strategy and token estimation",
        relevant_keywords=["chunk", "token", "split", "overlap", "estimate"],
        expected_files=["chunker.py"],
    ),
    RelevanceJudgment(
        query="Configuration and settings management",
        relevant_keywords=["config", "setting", "default", "weight", "parameter"],
        expected_files=["config.py"],
    ),
    RelevanceJudgment(
        query="Error handling and edge cases",
        relevant_keywords=["error", "exception", "try", "except", "raise", "validate"],
    ),
    RelevanceJudgment(
        query="Caching mechanism for queries",
        relevant_keywords=["cache", "ttl", "hit", "expire", "session"],
        expected_files=["db.py", "pipeline.py"],
    ),
    RelevanceJudgment(
        query="BM25 and full-text search implementation",
        relevant_keywords=["fts", "bm25", "match", "rank", "score"],
        expected_files=["retriever.py", "db.py"],
    ),
]


# Diverse query types for stress testing
STRESS_TEST_QUERIES = [
    # Short queries (1-3 words)
    "authentication",
    "error handling",
    "database",
    # Medium queries (4-7 words)
    "How does caching work here",
    "Find the main entry point",
    "What functions handle user input",
    # Long/complex queries
    "I need to understand how the retrieval pipeline combines BM25 and vector search",
    "What is the complete flow from user query to compressed context output",
    # Code-specific queries
    "def run_retrieval_pipeline",
    "class Candidate",
    "import sqlite3",
    # Natural language
    "explain the compression algorithm",
    "show me how embeddings are generated",
]


# ---------------------------------------------------------------------------
# Metrics Calculation
# ---------------------------------------------------------------------------

class QueryMetrics(TypedDict):
    """Metrics for a single query execution."""
    query: str
    latency_ms: float
    fts_hits: int
    vector_hits: int
    selected_chunks: int
    raw_tokens: int  # Tokens in selected chunks before compression
    compressed_tokens: int
    compression_ratio: float
    token_savings_pct: float
    precision_at_k: float
    recall_at_k: float
    ndcg: float
    mrr: float


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    inputs: list[Path]
    db_path: Path
    top_k: int = DEFAULT_TOP_K
    fts_k: int = DEFAULT_FTS_K
    vector_k: int = DEFAULT_VECTOR_K
    hybrid_mode: str = DEFAULT_HYBRID_MODE
    embedding_backend: str = DEFAULT_EMBEDDING_BACKEND
    embedding_model: str | None = DEFAULT_EMBEDDING_MODEL
    dimensions: int = DEFAULT_DIMENSIONS
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_CHUNK_OVERLAP
    word_budget: int = DEFAULT_WORD_BUDGET
    relevance_floor: float = DEFAULT_RELEVANCE_FLOOR
    num_iterations: int = 3  # Run each query multiple times for latency stats


def calculate_precision_at_k(
    retrieved_texts: list[str],
    judgment: RelevanceJudgment,
    k: int = 5,
) -> float:
    """Calculate Precision@K: fraction of top-K results that are relevant."""
    if not retrieved_texts or k == 0:
        return 0.0
    
    relevant_count = 0
    for text in retrieved_texts[:k]:
        text_lower = text.lower()
        # A result is relevant if it contains any of the relevant keywords
        if any(kw.lower() in text_lower for kw in judgment.relevant_keywords):
            # But not if it contains irrelevant keywords (higher weight)
            if not any(kw.lower() in text_lower for kw in judgment.irrelevant_keywords):
                relevant_count += 1
    
    return relevant_count / min(k, len(retrieved_texts))


def calculate_recall_at_k(
    retrieved_texts: list[str],
    judgment: RelevanceJudgment,
    k: int = 5,
) -> float:
    """Calculate Recall@K: fraction of relevant keywords found in top-K results."""
    if not judgment.relevant_keywords or k == 0:
        return 0.0
    
    combined_text = " ".join(retrieved_texts[:k]).lower()
    found_keywords = sum(
        1 for kw in judgment.relevant_keywords 
        if kw.lower() in combined_text
    )
    
    return found_keywords / len(judgment.relevant_keywords)


def calculate_mrr(
    retrieved_texts: list[str],
    judgment: RelevanceJudgment,
) -> float:
    """Calculate Mean Reciprocal Rank: 1/rank of first relevant result."""
    for i, text in enumerate(retrieved_texts, start=1):
        text_lower = text.lower()
        if any(kw.lower() in text_lower for kw in judgment.relevant_keywords):
            return 1.0 / i
    return 0.0


def calculate_ndcg(
    retrieved_texts: list[str],
    judgment: RelevanceJudgment,
    k: int = 5,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain."""
    import math
    
    if not retrieved_texts or k == 0:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, text in enumerate(retrieved_texts[:k], start=1):
        text_lower = text.lower()
        relevance = sum(1 for kw in judgment.relevant_keywords if kw.lower() in text_lower)
        # Normalize by keyword count to get 0-1 range
        norm_relevance = relevance / max(1, len(judgment.relevant_keywords))
        dcg += norm_relevance / math.log2(i + 1)
    
    # Calculate ideal DCG (all relevant at top)
    ideal_relevances = sorted(
        [1.0] * min(k, len(judgment.relevant_keywords)),
        reverse=True,
    )
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def run_single_query_benchmark(
    conn: sqlite3.Connection,
    db_path: Path,
    query: str,
    config: BenchmarkConfig,
    judgment: RelevanceJudgment | None = None,
) -> QueryMetrics:
    """Run benchmark for a single query and return metrics."""
    
    latencies: list[float] = []
    last_result = None
    
    for _ in range(config.num_iterations):
        start = time.perf_counter()
        result = run_retrieval_pipeline(
            conn=conn,
            db_path=db_path,
            query=query,
            top_k=config.top_k,
            fts_k=config.fts_k,
            vector_k=config.vector_k,
            min_fts_hits=DEFAULT_MIN_FTS_HITS,
            hybrid_mode=config.hybrid_mode,
            retrieval_mode="compact",
            embedding_backend=config.embedding_backend,
            embedding_model=config.embedding_model,
            session_id="benchmark",
            query_cache_ttl_seconds=0,  # Disable cache for fair measurement
            dimensions=config.dimensions,
            word_budget=config.word_budget,
            relevance_floor=config.relevance_floor,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        last_result = result
    
    if last_result is None:
        raise RuntimeError(f"No result for query: {query}")
    
    # Extract texts from candidates for accuracy measurement
    retrieved_texts = [c.source + " " + str(bullet) for c, bullet in 
                       zip(last_result.candidates, last_result.bullets, strict=False)]
    if not retrieved_texts:
        # Fallback to bullets only
        retrieved_texts = last_result.bullets
    
    # Calculate accuracy metrics if we have ground truth
    precision = 0.0
    recall = 0.0
    mrr = 0.0
    ndcg = 0.0
    
    if judgment:
        precision = calculate_precision_at_k(retrieved_texts, judgment, k=config.top_k)
        recall = calculate_recall_at_k(retrieved_texts, judgment, k=config.top_k)
        mrr = calculate_mrr(retrieved_texts, judgment)
        ndcg = calculate_ndcg(retrieved_texts, judgment, k=config.top_k)
    
    raw_tokens = last_result.token_metrics.selected_chunk_tokens
    compressed_tokens = last_result.token_metrics.compressed_tokens
    compression_ratio = raw_tokens / max(1, compressed_tokens)
    savings_pct = ((raw_tokens - compressed_tokens) / max(1, raw_tokens)) * 100
    
    return QueryMetrics(
        query=query,
        latency_ms=statistics.mean(latencies),
        fts_hits=last_result.retrieval.fts_hits,
        vector_hits=last_result.retrieval.vector_hits,
        selected_chunks=last_result.selected_chunks,
        raw_tokens=raw_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=round(compression_ratio, 2),
        token_savings_pct=round(savings_pct, 1),
        precision_at_k=round(precision, 3),
        recall_at_k=round(recall, 3),
        ndcg=round(ndcg, 3),
        mrr=round(mrr, 3),
    )


def calculate_baseline_tokens(file_paths: list[Path]) -> tuple[int, int]:
    """Calculate baseline token count if we sent everything (no reduction)."""
    total_tokens = 0
    total_chars = 0
    
    for path in file_paths:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            total_tokens += estimate_tokens(content)
            total_chars += len(content)
        except Exception:
            continue
    
    return total_tokens, total_chars


def run_comprehensive_benchmark(
    config: BenchmarkConfig,
    include_accuracy: bool = True,
    include_stress: bool = True,
) -> dict:
    """
    Run comprehensive benchmark suite and return detailed report.
    
    Returns a report with:
    - benchmark_summary: Overall statistics
    - baseline_comparison: Before/after token counts
    - latency_metrics: p50/p95/p99 latencies  
    - accuracy_metrics: Precision, Recall, MRR, NDCG
    - compression_metrics: Compression ratios
    - cost_analysis: Dollar savings estimates
    - query_details: Per-query breakdown
    """
    
    # Collect input files
    file_paths: list[Path] = []
    for input_path in config.inputs:
        if input_path.is_file():
            file_paths.append(input_path)
        elif input_path.is_dir():
            for ext in ["py", "ts", "js", "tsx", "jsx", "go", "rs", "java", "c", "cpp", "h", "md", "txt"]:
                file_paths.extend(input_path.rglob(f"*.{ext}"))
    
    if not file_paths:
        raise ValueError("No indexable files found")
    
    # Calculate baseline (sending everything)
    baseline_tokens, baseline_chars = calculate_baseline_tokens(file_paths)
    
    # Initialize database and index
    conn = connect_db(config.db_path)
    backend, model = resolve_embedding_backend(
        config.embedding_backend, config.embedding_model
    )
    
    # Index corpus
    index_start = time.perf_counter()
    stats = index_corpus(
        conn=conn,
        file_paths=[str(p) for p in file_paths],
        chunk_size_words=config.chunk_size,
        overlap_words=config.overlap,
        dimensions=config.dimensions,
        embedding_backend=backend,
        embedding_model=model,
    )
    index_latency_ms = (time.perf_counter() - index_start) * 1000
    
    # Run accuracy queries with ground truth
    accuracy_metrics: list[QueryMetrics] = []
    if include_accuracy:
        for judgment in GROUND_TRUTH_QUERIES:
            metrics = run_single_query_benchmark(
                conn=conn,
                db_path=config.db_path,
                query=judgment.query,
                config=config,
                judgment=judgment,
            )
            accuracy_metrics.append(metrics)
    
    # Run stress test queries
    stress_metrics: list[QueryMetrics] = []
    if include_stress:
        for query in STRESS_TEST_QUERIES:
            metrics = run_single_query_benchmark(
                conn=conn,
                db_path=config.db_path,
                query=query,
                config=config,
                judgment=None,
            )
            stress_metrics.append(metrics)
    
    conn.close()
    
    # Aggregate metrics
    all_metrics = accuracy_metrics + stress_metrics
    all_latencies = [m["latency_ms"] for m in all_metrics]
    all_latencies_sorted = sorted(all_latencies)
    
    # Calculate percentiles
    def percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]
    
    # Aggregate compression stats
    total_raw_tokens = sum(m["raw_tokens"] for m in all_metrics)
    total_compressed_tokens = sum(m["compressed_tokens"] for m in all_metrics)
    avg_compression_ratio = statistics.mean(m["compression_ratio"] for m in all_metrics) if all_metrics else 0
    avg_savings_pct = statistics.mean(m["token_savings_pct"] for m in all_metrics) if all_metrics else 0
    
    # Aggregate accuracy stats (only from ground truth queries)
    avg_precision = statistics.mean(m["precision_at_k"] for m in accuracy_metrics) if accuracy_metrics else 0
    avg_recall = statistics.mean(m["recall_at_k"] for m in accuracy_metrics) if accuracy_metrics else 0
    avg_mrr = statistics.mean(m["mrr"] for m in accuracy_metrics) if accuracy_metrics else 0
    avg_ndcg = statistics.mean(m["ndcg"] for m in accuracy_metrics) if accuracy_metrics else 0
    
    # Cost analysis (Claude Sonnet pricing as of 2024)
    # Input: $3 per million tokens
    cost_per_1m_tokens = 3.0
    baseline_cost_per_query = (baseline_tokens / 1_000_000) * cost_per_1m_tokens
    compressed_cost_per_query = (total_compressed_tokens / max(1, len(all_metrics)) / 1_000_000) * cost_per_1m_tokens
    savings_per_query = baseline_cost_per_query - compressed_cost_per_query
    
    report = {
        "benchmark_summary": {
            "files_indexed": len(file_paths),
            "chunks_created": stats["chunks_indexed"],
            "embedding_backend": backend,
            "embedding_model": model,
            "top_k": config.top_k,
            "hybrid_mode": config.hybrid_mode,
            "total_queries_run": len(all_metrics),
            "accuracy_queries": len(accuracy_metrics),
            "stress_queries": len(stress_metrics),
        },
        "baseline_comparison": {
            "baseline_tokens": baseline_tokens,
            "baseline_chars": baseline_chars,
            "avg_compressed_tokens_per_query": round(total_compressed_tokens / max(1, len(all_metrics)), 0),
            "reduction_vs_baseline_pct": round(
                ((baseline_tokens - (total_compressed_tokens / max(1, len(all_metrics)))) / max(1, baseline_tokens)) * 100, 1
            ),
            "interpretation": f"Sending entire codebase would cost {baseline_tokens:,} tokens. Token Reducer delivers ~{round(total_compressed_tokens / max(1, len(all_metrics))):,} tokens per query."
        },
        "latency_metrics": {
            "index_time_ms": round(index_latency_ms, 2),
            "p50_query_ms": round(percentile(all_latencies_sorted, 50), 2),
            "p95_query_ms": round(percentile(all_latencies_sorted, 95), 2),
            "p99_query_ms": round(percentile(all_latencies_sorted, 99), 2),
            "min_query_ms": round(min(all_latencies) if all_latencies else 0, 2),
            "max_query_ms": round(max(all_latencies) if all_latencies else 0, 2),
            "avg_query_ms": round(statistics.mean(all_latencies) if all_latencies else 0, 2),
        },
        "accuracy_metrics": {
            "precision_at_k": round(avg_precision, 3),
            "recall_at_k": round(avg_recall, 3),
            "mrr": round(avg_mrr, 3),
            "ndcg": round(avg_ndcg, 3),
            "interpretation": _interpret_accuracy(avg_precision, avg_recall, avg_mrr),
        },
        "compression_metrics": {
            "avg_compression_ratio": round(avg_compression_ratio, 2),
            "avg_token_savings_pct": round(avg_savings_pct, 1),
            "total_raw_tokens_retrieved": total_raw_tokens,
            "total_compressed_tokens": total_compressed_tokens,
        },
        "cost_analysis": {
            "pricing_model": "Claude Sonnet ($3/1M input tokens)",
            "baseline_cost_per_query_usd": round(baseline_cost_per_query, 4),
            "compressed_cost_per_query_usd": round(compressed_cost_per_query, 6),
            "savings_per_query_usd": round(savings_per_query, 4),
            "savings_per_100_queries_usd": round(savings_per_query * 100, 2),
            "savings_per_1000_queries_usd": round(savings_per_query * 1000, 2),
            "annual_savings_10_queries_day_usd": round(savings_per_query * 10 * 365, 2),
        },
        "accuracy_query_details": accuracy_metrics,
        "stress_query_details": stress_metrics,
    }
    
    return report


def _interpret_accuracy(precision: float, recall: float, mrr: float) -> str:
    """Generate human-readable interpretation of accuracy metrics."""
    if precision >= 0.8 and recall >= 0.7:
        quality = "Excellent"
        desc = "Retrieved chunks are highly relevant and cover most key concepts."
    elif precision >= 0.6 and recall >= 0.5:
        quality = "Good"
        desc = "Most retrieved chunks are relevant with reasonable coverage."
    elif precision >= 0.4 or recall >= 0.4:
        quality = "Fair"
        desc = "Some relevant chunks found, but coverage or precision could improve."
    else:
        quality = "Needs Improvement"
        desc = "Retrieved chunks may not be well-aligned with queries."
    
    return f"{quality}: {desc} (P@K={precision:.2f}, R@K={recall:.2f}, MRR={mrr:.2f})"


def format_benchmark_markdown(report: dict) -> str:
    """Format benchmark report as Markdown for documentation."""
    
    lines = [
        "# Token Reducer Benchmark Results",
        "",
        "## Summary",
        "",
        f"- **Files Indexed**: {report['benchmark_summary']['files_indexed']}",
        f"- **Chunks Created**: {report['benchmark_summary']['chunks_created']}",
        f"- **Embedding Backend**: {report['benchmark_summary']['embedding_backend']}",
        f"- **Queries Tested**: {report['benchmark_summary']['total_queries_run']}",
        "",
        "## Token Savings (Before vs After)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Baseline (full codebase) | {report['baseline_comparison']['baseline_tokens']:,} tokens |",
        f"| After Token Reducer | ~{report['baseline_comparison']['avg_compressed_tokens_per_query']:,.0f} tokens/query |",
        f"| **Reduction** | **{report['baseline_comparison']['reduction_vs_baseline_pct']}%** |",
        "",
        f"> {report['baseline_comparison']['interpretation']}",
        "",
        "## Latency Impact",
        "",
        "| Percentile | Latency |",
        "|------------|---------|",
        f"| p50 | {report['latency_metrics']['p50_query_ms']} ms |",
        f"| p95 | {report['latency_metrics']['p95_query_ms']} ms |",
        f"| p99 | {report['latency_metrics']['p99_query_ms']} ms |",
        f"| Index Time | {report['latency_metrics']['index_time_ms']} ms |",
        "",
        "## Accuracy Metrics",
        "",
        "| Metric | Score | Description |",
        "|--------|-------|-------------|",
        f"| Precision@K | {report['accuracy_metrics']['precision_at_k']} | Fraction of retrieved chunks that are relevant |",
        f"| Recall@K | {report['accuracy_metrics']['recall_at_k']} | Fraction of relevant concepts covered |",
        f"| MRR | {report['accuracy_metrics']['mrr']} | How quickly we find the first relevant result |",
        f"| NDCG | {report['accuracy_metrics']['ndcg']} | Quality of ranking (higher = better order) |",
        "",
        f"> **Assessment**: {report['accuracy_metrics']['interpretation']}",
        "",
        "## Cost Analysis",
        "",
        f"Based on {report['cost_analysis']['pricing_model']}:",
        "",
        "| Scenario | Cost |",
        "|----------|------|",
        f"| Without Token Reducer | ${report['cost_analysis']['baseline_cost_per_query_usd']:.4f}/query |",
        f"| With Token Reducer | ${report['cost_analysis']['compressed_cost_per_query_usd']:.6f}/query |",
        f"| **Savings per query** | **${report['cost_analysis']['savings_per_query_usd']:.4f}** |",
        f"| Savings per 100 queries | ${report['cost_analysis']['savings_per_100_queries_usd']:.2f} |",
        f"| Savings per 1,000 queries | ${report['cost_analysis']['savings_per_1000_queries_usd']:.2f} |",
        f"| Annual savings (10 queries/day) | ${report['cost_analysis']['annual_savings_10_queries_day_usd']:.2f} |",
        "",
        "## Compression Details",
        "",
        f"- **Average Compression Ratio**: {report['compression_metrics']['avg_compression_ratio']}x",
        f"- **Average Token Savings**: {report['compression_metrics']['avg_token_savings_pct']}%",
        "",
        "---",
        "",
        "*Generated by Token Reducer Benchmark Suite*",
    ]
    
    return "\n".join(lines)


def format_benchmark_json(report: dict) -> str:
    """Format benchmark report as JSON for CI/CD integration."""
    return json.dumps(report, indent=2)
