from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Ensure stdout is UTF-8 on Windows (avoids cp1252 UnicodeEncodeError for emoji/symbols)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from .config import (
    DEFAULT_ANN_ENGINE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DIMENSIONS,
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FTS_K,
    DEFAULT_HYBRID_MODE,
    DEFAULT_MIN_FTS_HITS,
    DEFAULT_QUERY_CACHE_TTL_SECONDS,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_K,
    DEFAULT_WORD_BUDGET,
    DEFAULT_DB_PATH,
)
from .db import connect_db, index_corpus
from .ann import build_hnsw_index, infer_embedding_dimensions
from .retriever import infer_retrieval_tier
from .compressor import build_packet, compress_candidates
from .pipeline import validate_query_input, run_retrieval_pipeline
from .chunker import (
    collect_input_files,
    clean_text,
    normalize_chunking,
    estimate_tokens,
)
from .embeddings import resolve_embedding_backend
from .db import upsert_document


def cmd_index(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser().resolve()
    conn = connect_db(db_path)
    try:
        files = collect_input_files(args.inputs)
        if not files:
            print("No indexable files found.", file=sys.stderr)
            return 1

        embedding_backend, embedding_model = resolve_embedding_backend(
            requested_backend=args.embedding_backend,
            requested_model=args.embedding_model,
        )
        chunk_size_words, overlap_words = normalize_chunking(args.chunk_size, args.overlap)

        stats = index_corpus(
            conn=conn,
            file_paths=files,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            dimensions=args.dimensions,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
        )

        inferred_dims = infer_embedding_dimensions(
            conn=conn,
            backend=embedding_backend,
            model_name=embedding_model,
            fallback_dimensions=args.dimensions,
        )
        ann_built = build_hnsw_index(
            conn=conn,
            db_path=db_path,
            backend=embedding_backend,
            dimensions=inferred_dims,
            model_name=embedding_model,
        )

        print(
            json.dumps(
                {
                    "db": str(db_path),
                    "files_indexed": stats["files_indexed"],
                    "chunks_indexed": stats["chunks_indexed"],
                    "embedding_backend": embedding_backend,
                    "embedding_model": embedding_model,
                    "ann_engine": DEFAULT_ANN_ENGINE,
                    "ann_index_built": ann_built,
                },
                indent=2,
            )
        )
        return 0
    finally:
        conn.close()


def cmd_query(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser().resolve()
    conn = connect_db(db_path)
    try:
        query_ok, query_error = validate_query_input(args.query)
        if not query_ok:
            print(query_error, file=sys.stderr)
            return 2

        embedding_backend, embedding_model = resolve_embedding_backend(
            requested_backend=args.embedding_backend,
            requested_model=args.embedding_model,
        )

        result = run_retrieval_pipeline(
            conn=conn,
            db_path=db_path,
            query=args.query,
            top_k=args.top_k,
            fts_k=args.fts_k,
            vector_k=args.vector_k,
            min_fts_hits=args.min_fts_hits,
            hybrid_mode=args.hybrid_mode,
            retrieval_mode=args.mode,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            session_id=args.session_id,
            query_cache_ttl_seconds=args.query_cache_ttl,
            dimensions=args.dimensions,
            word_budget=args.word_budget,
        )

        print(json.dumps(result, indent=2) if args.json else result["packet"])
        return 0
    finally:
        conn.close()


def cmd_run(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser().resolve()
    conn = connect_db(db_path)
    try:
        query_ok, query_error = validate_query_input(args.query)
        if not query_ok:
            print(query_error, file=sys.stderr)
            return 2

        embedding_backend, embedding_model = resolve_embedding_backend(
            requested_backend=args.embedding_backend,
            requested_model=args.embedding_model,
        )

        chunk_size_words, overlap_words = normalize_chunking(args.chunk_size, args.overlap)

        if args.inputs:
            files = collect_input_files(args.inputs)
            if not files:
                print("No indexable files found for run mode.", file=sys.stderr)
                return 1
            index_corpus(
                conn=conn,
                file_paths=files,
                chunk_size_words=chunk_size_words,
                overlap_words=overlap_words,
                dimensions=args.dimensions,
                embedding_backend=embedding_backend,
                embedding_model=embedding_model,
            )
            # Only build ANN index for large codebases; it's unnecessary overhead
            # for small/medium projects and significantly slows down the pipeline.
            adaptive_tier = infer_retrieval_tier(conn)
            if adaptive_tier == "full_hybrid":
                inferred_dims = infer_embedding_dimensions(
                    conn=conn,
                    backend=embedding_backend,
                    model_name=embedding_model,
                    fallback_dimensions=args.dimensions,
                )
                build_hnsw_index(
                    conn=conn,
                    db_path=db_path,
                    backend=embedding_backend,
                    dimensions=inferred_dims,
                    model_name=embedding_model,
                )
            else:
                print(
                    f"[info] Adaptive tier '{adaptive_tier}' — skipping ANN index build (FTS5 is sufficient).",
                    file=sys.stderr,
                )

        result = run_retrieval_pipeline(
            conn=conn,
            db_path=db_path,
            query=args.query,
            top_k=args.top_k,
            fts_k=args.fts_k,
            vector_k=args.vector_k,
            min_fts_hits=args.min_fts_hits,
            hybrid_mode=args.hybrid_mode,
            retrieval_mode=args.mode,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            session_id=args.session_id,
            query_cache_ttl_seconds=args.query_cache_ttl,
            dimensions=args.dimensions,
            word_budget=args.word_budget,
        )

        print(json.dumps(result, indent=2) if args.json else result["packet"])
        return 0
    finally:
        conn.close()


def cmd_compress_raw(args: argparse.Namespace) -> int:
    """Zero-Turn compression: read raw text from stdin, compress it, output the packet.

    Used by userprompt_guard.py to transparently intercept large prompts before
    they reach the LLM context window (Transparent Pre-Compression Hook / TPCH).
    """
    raw_text = sys.stdin.read()
    if not raw_text.strip():
        return 0

    db_path = Path(":memory:")
    conn = connect_db(db_path)
    try:
        chunk_size_words, overlap_words = normalize_chunking(args.chunk_size, args.overlap)
        cleaned = clean_text(raw_text)
        upsert_document(
            conn=conn,
            source="intercepted_prompt.txt",
            raw_text=raw_text,
            cleaned_text=cleaned,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            dimensions=args.dimensions,
            embedding_backend="hash",
            embedding_model=None,
        )
        conn.commit()

        query = args.query if args.query.strip() else " ".join(raw_text.split()[:50])

        result = run_retrieval_pipeline(
            conn=conn,
            db_path=db_path,
            query=query,
            top_k=args.top_k,
            fts_k=args.fts_k,
            vector_k=0,
            min_fts_hits=1,
            hybrid_mode="fallback",
            retrieval_mode="compact",
            embedding_backend="hash",
            embedding_model=None,
            session_id="auto",
            query_cache_ttl_seconds=0,
            dimensions=args.dimensions,
            word_budget=args.word_budget,
        )

        print(result["packet"])
        return 0
    finally:
        conn.close()


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run comprehensive benchmarks and generate metrics report."""
    import time

    db_path = Path(args.db).expanduser().resolve()

    # Collect test files
    files = collect_input_files(args.inputs) if args.inputs else []
    if not files:
        print("No input files for benchmarking. Provide --inputs.", file=sys.stderr)
        return 1

    # Calculate raw input stats
    raw_tokens_total = 0
    raw_chars_total = 0
    file_stats: list[dict] = []

    for file_path in files:
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="replace")
            tokens = estimate_tokens(content)
            raw_tokens_total += tokens
            raw_chars_total += len(content)
            file_stats.append({
                "file": str(file_path),
                "tokens": tokens,
                "chars": len(content),
            })
        except Exception:
            continue

    # Benchmark indexing
    conn = connect_db(db_path)
    embedding_backend, embedding_model = resolve_embedding_backend(
        requested_backend=args.embedding_backend,
        requested_model=args.embedding_model,
    )
    chunk_size_words, overlap_words = normalize_chunking(args.chunk_size, args.overlap)

    index_start = time.perf_counter()
    stats = index_corpus(
        conn=conn,
        file_paths=files,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
        dimensions=args.dimensions,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
    )
    index_latency_ms = (time.perf_counter() - index_start) * 1000

    # Build ANN index
    inferred_dims = infer_embedding_dimensions(
        conn=conn,
        backend=embedding_backend,
        model_name=embedding_model,
        fallback_dimensions=args.dimensions,
    )
    build_hnsw_index(
        conn=conn,
        db_path=db_path,
        backend=embedding_backend,
        dimensions=inferred_dims,
        model_name=embedding_model,
    )

    # Benchmark queries
    test_queries = [
        "How does authentication work?",
        "What functions handle database connections?",
        "Error handling patterns in this codebase",
        "Main entry point and initialization",
        "API endpoint implementations",
    ]

    query_metrics: list[dict] = []
    compressed_tokens_total = 0

    for query in test_queries:
        query_start = time.perf_counter()
        result = run_retrieval_pipeline(
            conn=conn,
            db_path=db_path,
            query=query,
            top_k=args.top_k,
            fts_k=args.fts_k,
            vector_k=args.vector_k,
            min_fts_hits=DEFAULT_MIN_FTS_HITS,
            hybrid_mode=args.hybrid_mode,
            retrieval_mode="compact",
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            session_id="benchmark",
            query_cache_ttl_seconds=0,  # Disable cache for benchmarking
            dimensions=args.dimensions,
            word_budget=args.word_budget,
        )
        query_latency_ms = (time.perf_counter() - query_start) * 1000

        compressed_tokens = result.get("token_metrics", {}).get("compressed_tokens", 0)
        selected_tokens = result.get("token_metrics", {}).get("selected_chunk_tokens", 0)
        compressed_tokens_total += compressed_tokens

        query_metrics.append({
            "query": query,
            "latency_ms": round(query_latency_ms, 2),
            "fts_hits": result.get("retrieval", {}).get("fts_hits", 0),
            "vector_hits": result.get("retrieval", {}).get("vector_hits", 0),
            "selected_chunks": result.get("selected_chunks", 0),
            "selected_tokens": selected_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": round(selected_tokens / max(1, compressed_tokens), 2),
        })

    conn.close()

    # Calculate aggregate metrics
    avg_query_latency = sum(q["latency_ms"] for q in query_metrics) / len(query_metrics)
    avg_compression_ratio = sum(q["compression_ratio"] for q in query_metrics) / len(query_metrics)

    # Token savings analysis
    # If user sent all raw content vs using token-reducer
    hypothetical_saved = raw_tokens_total - (compressed_tokens_total / len(test_queries))
    savings_pct = (hypothetical_saved / raw_tokens_total * 100) if raw_tokens_total > 0 else 0

    # Cost savings estimate (assuming $0.01 per 1K input tokens for Claude)
    cost_per_1k_tokens = 0.01
    raw_cost = (raw_tokens_total / 1000) * cost_per_1k_tokens
    compressed_cost = (compressed_tokens_total / len(test_queries) / 1000) * cost_per_1k_tokens
    cost_savings = raw_cost - compressed_cost

    report = {
        "benchmark_summary": {
            "files_indexed": len(files),
            "chunks_created": stats["chunks_indexed"],
            "raw_input_tokens": raw_tokens_total,
            "raw_input_chars": raw_chars_total,
            "embedding_backend": embedding_backend,
            "embedding_model": embedding_model,
        },
        "latency_metrics": {
            "index_time_ms": round(index_latency_ms, 2),
            "avg_query_time_ms": round(avg_query_latency, 2),
            "min_query_time_ms": round(min(q["latency_ms"] for q in query_metrics), 2),
            "max_query_time_ms": round(max(q["latency_ms"] for q in query_metrics), 2),
        },
        "compression_metrics": {
            "avg_compression_ratio": round(avg_compression_ratio, 2),
            "avg_compressed_tokens_per_query": round(compressed_tokens_total / len(test_queries), 0),
            "total_queries_benchmarked": len(test_queries),
        },
        "token_savings": {
            "raw_context_tokens": raw_tokens_total,
            "avg_compressed_tokens": round(compressed_tokens_total / len(test_queries), 0),
            "estimated_savings_pct": round(savings_pct, 1),
            "tokens_saved_per_query": round(hypothetical_saved, 0),
        },
        "cost_analysis": {
            "note": "Estimated at $0.01 per 1K input tokens",
            "raw_context_cost_usd": round(raw_cost, 4),
            "compressed_cost_usd": round(compressed_cost, 4),
            "savings_per_query_usd": round(cost_savings, 4),
            "savings_per_100_queries_usd": round(cost_savings * 100, 2),
        },
        "query_details": query_metrics,
    }

    print(json.dumps(report, indent=2))

    # Print human-readable summary
    print("\n" + "=" * 60, file=sys.stderr)
    print("TOKEN REDUCER BENCHMARK SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Files indexed: {len(files)}", file=sys.stderr)
    print(f"Raw input tokens: {raw_tokens_total:,}", file=sys.stderr)
    print(f"Avg compressed tokens: {compressed_tokens_total // len(test_queries):,}", file=sys.stderr)
    print(f"Token savings: {savings_pct:.1f}%", file=sys.stderr)
    print(f"Avg query latency: {avg_query_latency:.1f}ms", file=sys.stderr)
    print(f"Cost savings per 100 queries: ${cost_savings * 100:.2f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    return 0


def cmd_self_test(_args: argparse.Namespace) -> int:
    from .db import index_corpus as _index_corpus
    from .chunker import collect_input_files as _collect_input_files

    with tempfile.TemporaryDirectory(prefix="token-reducer-self-test-") as tmp_dir:
        root = Path(tmp_dir)
        db_path = root / "index.db"

        sample_files = {
            "auth_middleware.md": """
            The authentication middleware checks JWT tokens in cookies.
            If the token is invalid, the middleware redirects to /login.
            It runs before protected route handlers.
            """,
            "session_guard.md": """
            Authorization guard validates session credentials prior to route execution.
            The guard denies unauthorized access and records audit logs.
            """,
            "unrelated_notes.md": """
            CSS palette and typography notes for landing page.
            Icon sizing and spacing guidance for components.
            """,
        }

        for name, content in sample_files.items():
            (root / name).write_text(content.strip() + "\n", encoding="utf-8")

        conn = connect_db(db_path)
        try:
            files = _collect_input_files([str(root)])
            _index_corpus(
                conn=conn,
                file_paths=files,
                chunk_size_words=DEFAULT_CHUNK_SIZE,
                overlap_words=DEFAULT_CHUNK_OVERLAP,
                dimensions=DEFAULT_DIMENSIONS,
                embedding_backend="hash",
                embedding_model=None,
            )
            result = run_retrieval_pipeline(
                conn=conn,
                db_path=db_path,
                query="How does middleware validate auth tokens before protected routes?",
                top_k=5,
                fts_k=8,
                vector_k=8,
                min_fts_hits=DEFAULT_MIN_FTS_HITS,
                hybrid_mode="always",
                retrieval_mode="compact",
                embedding_backend="hash",
                embedding_model=None,
                session_id="self-test",
                query_cache_ttl_seconds=DEFAULT_QUERY_CACHE_TTL_SECONDS,
                dimensions=DEFAULT_DIMENSIONS,
                word_budget=DEFAULT_WORD_BUDGET,
            )

            result_repeat = run_retrieval_pipeline(
                conn=conn,
                db_path=db_path,
                query="How does middleware validate auth tokens before protected routes?",
                top_k=5,
                fts_k=8,
                vector_k=8,
                min_fts_hits=DEFAULT_MIN_FTS_HITS,
                hybrid_mode="always",
                retrieval_mode="compact",
                embedding_backend="hash",
                embedding_model=None,
                session_id="self-test",
                query_cache_ttl_seconds=DEFAULT_QUERY_CACHE_TTL_SECONDS,
                dimensions=DEFAULT_DIMENSIONS,
                word_budget=DEFAULT_WORD_BUDGET,
            )
        finally:
            conn.close()

    checks = {
        "bm25_enabled": bool(result["retrieval"]["bm25_enabled"]),
        "fts_hits_gt_zero": int(result["retrieval"]["fts_hits"]) > 0,
        "vector_hits_gt_zero": int(result["retrieval"]["vector_hits"]) > 0,
        "selected_chunks_le_5": int(result["selected_chunks"]) <= 5,
        "compressed_le_selected_tokens": int(result["token_metrics"]["compressed_tokens"])
        <= int(result["token_metrics"]["selected_chunk_tokens"]),
        "cache_hit_on_repeat": bool(result_repeat.get("cache", {}).get("hit", False)),
        "session_memory_present": bool(result_repeat.get("session_memory", {}).get("recent_queries")),
    }

    ok = all(checks.values())
    summary = {
        "ok": ok,
        "checks": checks,
        "retrieval": result["retrieval"],
        "token_metrics": result["token_metrics"],
        "repeat_query_cache": result_repeat.get("cache", {}),
    }
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="context_pipeline",
        description="Local-first context slimming pipeline for Claude Code.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Preprocess and index input files.")
    index_parser.add_argument("--inputs", nargs="+", required=True, help="Files or directories to index.")
    index_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite index path.")
    index_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    index_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    index_parser.add_argument("--embedding-backend", choices=["hash", "ml"], default=DEFAULT_EMBEDDING_BACKEND)
    index_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    index_parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)

    query_parser = subparsers.add_parser("query", help="Retrieve and compress from existing index.")
    query_parser.add_argument("--query", required=True, help="Question or objective.")
    query_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite index path.")
    query_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    query_parser.add_argument("--fts-k", type=int, default=DEFAULT_FTS_K)
    query_parser.add_argument("--vector-k", type=int, default=DEFAULT_VECTOR_K)
    query_parser.add_argument("--min-fts-hits", type=int, default=DEFAULT_MIN_FTS_HITS)
    query_parser.add_argument("--mode", choices=["compact", "deep-debug"], default=DEFAULT_RETRIEVAL_MODE)
    query_parser.add_argument("--hybrid-mode", choices=["always", "fallback"], default=DEFAULT_HYBRID_MODE)
    query_parser.add_argument("--embedding-backend", choices=["hash", "ml"], default=DEFAULT_EMBEDDING_BACKEND)
    query_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    query_parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    query_parser.add_argument("--session-id", default="default")
    query_parser.add_argument("--query-cache-ttl", type=int, default=DEFAULT_QUERY_CACHE_TTL_SECONDS)
    query_parser.add_argument("--word-budget", type=int, default=DEFAULT_WORD_BUDGET)
    query_parser.add_argument("--json", action="store_true")

    run_parser = subparsers.add_parser("run", help="End-to-end run: optional indexing, then retrieval and compression.")
    run_parser.add_argument("--inputs", nargs="*", default=[], help="Files or directories to index first.")
    run_parser.add_argument("--query", required=True, help="Question or objective.")
    run_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite index path.")
    run_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    run_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    run_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    run_parser.add_argument("--fts-k", type=int, default=DEFAULT_FTS_K)
    run_parser.add_argument("--vector-k", type=int, default=DEFAULT_VECTOR_K)
    run_parser.add_argument("--min-fts-hits", type=int, default=DEFAULT_MIN_FTS_HITS)
    run_parser.add_argument("--mode", choices=["compact", "deep-debug"], default=DEFAULT_RETRIEVAL_MODE)
    run_parser.add_argument("--hybrid-mode", choices=["always", "fallback"], default=DEFAULT_HYBRID_MODE)
    run_parser.add_argument("--embedding-backend", choices=["hash", "ml"], default=DEFAULT_EMBEDDING_BACKEND)
    run_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    run_parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    run_parser.add_argument("--session-id", default="default")
    run_parser.add_argument("--query-cache-ttl", type=int, default=DEFAULT_QUERY_CACHE_TTL_SECONDS)
    run_parser.add_argument("--word-budget", type=int, default=DEFAULT_WORD_BUDGET)
    run_parser.add_argument("--json", action="store_true")

    compress_raw_parser = subparsers.add_parser(
        "compress-raw",
        help="Zero-Turn: read raw text from stdin, compress it in-memory, and output the context packet.",
    )
    compress_raw_parser.add_argument("--query", default="", help="Optional query to focus compression")
    compress_raw_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    compress_raw_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    compress_raw_parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    compress_raw_parser.add_argument("--top-k", type=int, default=5)
    compress_raw_parser.add_argument("--fts-k", type=int, default=12)
    compress_raw_parser.add_argument("--word-budget", type=int, default=500)

    subparsers.add_parser("self-test", help="Run internal checks for BM25+hybrid retrieval and token reduction behavior.")

    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks and generate metrics report.")
    benchmark_parser.add_argument("--inputs", nargs="+", required=True, help="Files or directories to benchmark.")
    benchmark_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite index path.")
    benchmark_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    benchmark_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    benchmark_parser.add_argument("--embedding-backend", choices=["hash", "ml"], default=DEFAULT_EMBEDDING_BACKEND)
    benchmark_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    benchmark_parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    benchmark_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    benchmark_parser.add_argument("--fts-k", type=int, default=DEFAULT_FTS_K)
    benchmark_parser.add_argument("--vector-k", type=int, default=DEFAULT_VECTOR_K)
    benchmark_parser.add_argument("--hybrid-mode", choices=["always", "fallback"], default=DEFAULT_HYBRID_MODE)
    benchmark_parser.add_argument("--word-budget", type=int, default=DEFAULT_WORD_BUDGET)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        return cmd_index(args)
    if args.command == "query":
        return cmd_query(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "compress-raw":
        return cmd_compress_raw(args)
    if args.command == "self-test":
        return cmd_self_test(args)
    if args.command == "benchmark":
        return cmd_benchmark(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
