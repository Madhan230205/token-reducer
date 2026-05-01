from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Ensure stdout is UTF-8 on Windows (avoids cp1252 UnicodeEncodeError for emoji/symbols)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from .ann import build_hnsw_index, infer_embedding_dimensions
from .chunker import (
    clean_text,
    collect_input_files,
    estimate_tokens,
    normalize_chunking,
)
from .config import (
    DEFAULT_ANN_ENGINE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DB_PATH,
    DEFAULT_DIMENSIONS,
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FTS_K,
    DEFAULT_HYBRID_MODE,
    DEFAULT_MIN_FTS_HITS,
    DEFAULT_QUERY_CACHE_TTL_SECONDS,
    DEFAULT_RELEVANCE_FLOOR,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_VECTOR_K,
    DEFAULT_WORD_BUDGET,
)
from .db import (
    connect_db,
    detect_file_changes,
    garbage_collect,
    get_database_size,
    get_index_stats,
    index_corpus,
    remove_documents_by_source,
    update_document_mtime,
    upsert_document,
    vacuum_database,
)
from .embeddings import resolve_embedding_backend
from .pipeline import run_retrieval_pipeline, validate_query_input
from .plugin_settings import get_runtime_defaults
from .retriever import infer_retrieval_tier

_RUNTIME = get_runtime_defaults()

app = typer.Typer(
    name="token-reducer",
    help="Local-first context slimming pipeline for Claude Code — cuts token usage without losing signal.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
err = Console(stderr=True)

# ---------------------------------------------------------------------------
# Shared option types
# ---------------------------------------------------------------------------

DbOpt = Annotated[str, typer.Option("--db", help="SQLite index path.")]
ChunkSizeOpt = Annotated[int, typer.Option("--chunk-size")]
OverlapOpt = Annotated[int, typer.Option("--overlap")]
TopKOpt = Annotated[int, typer.Option("--top-k")]
FtsKOpt = Annotated[int, typer.Option("--fts-k")]
VectorKOpt = Annotated[int, typer.Option("--vector-k")]
MinFtsHitsOpt = Annotated[int, typer.Option("--min-fts-hits")]
HybridModeOpt = Annotated[str, typer.Option("--hybrid-mode", help="always | fallback")]
EmbBackendOpt = Annotated[str, typer.Option("--embedding-backend", help="hash | ml")]
EmbModelOpt = Annotated[str | None, typer.Option("--embedding-model")]
DimensionsOpt = Annotated[int, typer.Option("--dimensions")]
SessionOpt = Annotated[str, typer.Option("--session-id")]
CacheTtlOpt = Annotated[int, typer.Option("--query-cache-ttl")]
WordBudgetOpt = Annotated[int, typer.Option("--word-budget")]
RelevanceFloorOpt = Annotated[
    float,
    typer.Option(
        "--relevance-floor",
        help="Minimum final_score for a chunk to be summarized (higher drops more).",
    ),
]
ModeOpt = Annotated[str, typer.Option("--mode", help="compact | deep-debug")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Output full JSON result.")]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def index(
    inputs: Annotated[list[str], typer.Option("--inputs", help="Files or directories to index.")],
    db: DbOpt = str(DEFAULT_DB_PATH),
    chunk_size: ChunkSizeOpt = _RUNTIME.chunk_size_words,
    overlap: OverlapOpt = _RUNTIME.chunk_overlap_words,
    embedding_backend: EmbBackendOpt = DEFAULT_EMBEDDING_BACKEND,
    embedding_model: EmbModelOpt = DEFAULT_EMBEDDING_MODEL,
    dimensions: DimensionsOpt = DEFAULT_DIMENSIONS,
) -> None:
    """Preprocess and index input files into the local SQLite store."""
    db_path = Path(db).expanduser().resolve()
    conn = connect_db(db_path)
    try:
        files = collect_input_files(inputs)
        if not files:
            err.print("[red]No indexable files found.[/red]")
            raise typer.Exit(1)

        backend, model = resolve_embedding_backend(
            requested_backend=embedding_backend,
            requested_model=embedding_model,
        )
        chunk_size_words, overlap_words = normalize_chunking(chunk_size, overlap)

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=err) as progress:
            progress.add_task(f"Indexing {len(files)} file(s)…")
            stats = index_corpus(
                conn=conn,
                file_paths=files,
                chunk_size_words=chunk_size_words,
                overlap_words=overlap_words,
                dimensions=dimensions,
                embedding_backend=backend,
                embedding_model=model,
            )

        inferred_dims = infer_embedding_dimensions(
            conn=conn, backend=backend, model_name=model, fallback_dimensions=dimensions
        )
        ann_built = build_hnsw_index(
            conn=conn, db_path=db_path, backend=backend, dimensions=inferred_dims, model_name=model
        )

        typer.echo(
            json.dumps(
                {
                    "db": str(db_path),
                    "files_indexed": stats["files_indexed"],
                    "chunks_indexed": stats["chunks_indexed"],
                    "embedding_backend": backend,
                    "embedding_model": model,
                    "ann_engine": DEFAULT_ANN_ENGINE,
                    "ann_index_built": ann_built,
                },
                indent=2,
            )
        )
    finally:
        conn.close()


@app.command()
def query(
    query_text: Annotated[str, typer.Option("--query", help="Question or objective.")],
    db: DbOpt = str(DEFAULT_DB_PATH),
    top_k: TopKOpt = _RUNTIME.default_top_k,
    fts_k: FtsKOpt = DEFAULT_FTS_K,
    vector_k: VectorKOpt = DEFAULT_VECTOR_K,
    min_fts_hits: MinFtsHitsOpt = DEFAULT_MIN_FTS_HITS,
    mode: ModeOpt = DEFAULT_RETRIEVAL_MODE,
    hybrid_mode: HybridModeOpt = DEFAULT_HYBRID_MODE,
    embedding_backend: EmbBackendOpt = DEFAULT_EMBEDDING_BACKEND,
    embedding_model: EmbModelOpt = DEFAULT_EMBEDDING_MODEL,
    dimensions: DimensionsOpt = DEFAULT_DIMENSIONS,
    session_id: SessionOpt = "default",
    query_cache_ttl: CacheTtlOpt = DEFAULT_QUERY_CACHE_TTL_SECONDS,
    word_budget: WordBudgetOpt = _RUNTIME.compression_word_budget,
    relevance_floor: RelevanceFloorOpt = _RUNTIME.relevance_floor,
    output_json: JsonOpt = False,
) -> None:
    """Retrieve and compress context from an existing index."""
    db_path = Path(db).expanduser().resolve()
    conn = connect_db(db_path)
    try:
        query_ok, query_error = validate_query_input(query_text)
        if not query_ok:
            err.print(f"[red]{query_error}[/red]")
            raise typer.Exit(2)

        backend, model = resolve_embedding_backend(
            requested_backend=embedding_backend, requested_model=embedding_model
        )

        result = run_retrieval_pipeline(
            conn=conn,
            db_path=db_path,
            query=query_text,
            top_k=top_k,
            fts_k=fts_k,
            vector_k=vector_k,
            min_fts_hits=min_fts_hits,
            hybrid_mode=hybrid_mode,
            retrieval_mode=mode,
            embedding_backend=backend,
            embedding_model=model,
            session_id=session_id,
            query_cache_ttl_seconds=query_cache_ttl,
            dimensions=dimensions,
            word_budget=word_budget,
            relevance_floor=relevance_floor,
        )

        typer.echo(json.dumps(result.model_dump(), indent=2) if output_json else result.packet)
    finally:
        conn.close()


@app.command()
def run(
    query_text: Annotated[str, typer.Option("--query", help="Question or objective.")],
    inputs: Annotated[
        list[str], typer.Option("--inputs", help="Files or directories to index first.")
    ] = [],  # noqa: B006
    db: DbOpt = str(DEFAULT_DB_PATH),
    chunk_size: ChunkSizeOpt = _RUNTIME.chunk_size_words,
    overlap: OverlapOpt = _RUNTIME.chunk_overlap_words,
    top_k: TopKOpt = _RUNTIME.default_top_k,
    fts_k: FtsKOpt = DEFAULT_FTS_K,
    vector_k: VectorKOpt = DEFAULT_VECTOR_K,
    min_fts_hits: MinFtsHitsOpt = DEFAULT_MIN_FTS_HITS,
    mode: ModeOpt = DEFAULT_RETRIEVAL_MODE,
    hybrid_mode: HybridModeOpt = DEFAULT_HYBRID_MODE,
    embedding_backend: EmbBackendOpt = DEFAULT_EMBEDDING_BACKEND,
    embedding_model: EmbModelOpt = DEFAULT_EMBEDDING_MODEL,
    dimensions: DimensionsOpt = DEFAULT_DIMENSIONS,
    session_id: SessionOpt = "default",
    query_cache_ttl: CacheTtlOpt = DEFAULT_QUERY_CACHE_TTL_SECONDS,
    word_budget: WordBudgetOpt = _RUNTIME.compression_word_budget,
    relevance_floor: RelevanceFloorOpt = _RUNTIME.relevance_floor,
    output_json: JsonOpt = False,
) -> None:
    """End-to-end: optionally index files, then retrieve and compress."""
    db_path = Path(db).expanduser().resolve()
    conn = connect_db(db_path)
    try:
        query_ok, query_error = validate_query_input(query_text)
        if not query_ok:
            err.print(f"[red]{query_error}[/red]")
            raise typer.Exit(2)

        backend, model = resolve_embedding_backend(
            requested_backend=embedding_backend, requested_model=embedding_model
        )
        chunk_size_words, overlap_words = normalize_chunking(chunk_size, overlap)

        if inputs:
            files = collect_input_files(inputs)
            if not files:
                err.print("[red]No indexable files found for run mode.[/red]")
                raise typer.Exit(1)

            with Progress(
                SpinnerColumn(), TextColumn("{task.description}"), console=err
            ) as progress:
                progress.add_task(f"Indexing {len(files)} file(s)…")
                index_corpus(
                    conn=conn,
                    file_paths=files,
                    chunk_size_words=chunk_size_words,
                    overlap_words=overlap_words,
                    dimensions=dimensions,
                    embedding_backend=backend,
                    embedding_model=model,
                )

            adaptive_tier = infer_retrieval_tier(conn)
            if adaptive_tier == "full_hybrid":
                inferred_dims = infer_embedding_dimensions(
                    conn=conn, backend=backend, model_name=model, fallback_dimensions=dimensions
                )
                build_hnsw_index(
                    conn=conn,
                    db_path=db_path,
                    backend=backend,
                    dimensions=inferred_dims,
                    model_name=model,
                )
            else:
                err.print(
                    f"[dim][info] Adaptive tier '{adaptive_tier}' \u2014 skipping ANN index build (FTS5 is sufficient).[/dim]"
                )

        result = run_retrieval_pipeline(
            conn=conn,
            db_path=db_path,
            query=query_text,
            top_k=top_k,
            fts_k=fts_k,
            vector_k=vector_k,
            min_fts_hits=min_fts_hits,
            hybrid_mode=hybrid_mode,
            retrieval_mode=mode,
            embedding_backend=backend,
            embedding_model=model,
            session_id=session_id,
            query_cache_ttl_seconds=query_cache_ttl,
            dimensions=dimensions,
            word_budget=word_budget,
            relevance_floor=relevance_floor,
        )

        typer.echo(json.dumps(result.model_dump(), indent=2) if output_json else result.packet)
    finally:
        conn.close()


@app.command("compress-raw")
def compress_raw(
    query_text: Annotated[
        str, typer.Option("--query", help="Optional query to focus compression.")
    ] = "",
    chunk_size: ChunkSizeOpt = _RUNTIME.chunk_size_words,
    overlap: OverlapOpt = _RUNTIME.chunk_overlap_words,
    dimensions: DimensionsOpt = DEFAULT_DIMENSIONS,
    top_k: TopKOpt = _RUNTIME.default_top_k,
    fts_k: FtsKOpt = 12,
    word_budget: WordBudgetOpt = 500,
    relevance_floor: RelevanceFloorOpt = _RUNTIME.relevance_floor,
) -> None:
    """Zero-Turn compression: read raw text from stdin and output a context packet.

    Used by userprompt_guard.py (Transparent Pre-Compression Hook / TPCH).
    """
    raw_text = sys.stdin.read()
    if not raw_text.strip():
        return

    db_path = Path(":memory:")
    conn = connect_db(db_path)
    try:
        chunk_size_words, overlap_words = normalize_chunking(chunk_size, overlap)
        cleaned = clean_text(raw_text)
        upsert_document(
            conn=conn,
            source="intercepted_prompt.txt",
            raw_text=raw_text,
            cleaned_text=cleaned,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            dimensions=dimensions,
            embedding_backend="hash",
            embedding_model=None,
        )
        conn.commit()

        effective_query = query_text.strip() or " ".join(raw_text.split()[:50])

        result = run_retrieval_pipeline(
            conn=conn,
            db_path=db_path,
            query=effective_query,
            top_k=top_k,
            fts_k=fts_k,
            vector_k=0,
            min_fts_hits=1,
            hybrid_mode="fallback",
            retrieval_mode="compact",
            embedding_backend="hash",
            embedding_model=None,
            session_id="auto",
            query_cache_ttl_seconds=0,
            dimensions=dimensions,
            word_budget=word_budget,
            relevance_floor=relevance_floor,
        )

        typer.echo(result.packet)
    finally:
        conn.close()


@app.command("self-test")
def self_test() -> None:
    """Run internal checks for BM25+hybrid retrieval and token reduction behaviour."""
    from .chunker import collect_input_files as _collect_input_files
    from .db import index_corpus as _index_corpus

    with tempfile.TemporaryDirectory(prefix="token-reducer-self-test-") as tmp_dir:
        root = Path(tmp_dir)
        db_path = root / "index.db"

        sample_files = {
            "auth_middleware.md": (
                "The authentication middleware checks JWT tokens in cookies.\n"
                "If the token is invalid, the middleware redirects to /login.\n"
                "It runs before protected route handlers.\n"
            ),
            "session_guard.md": (
                "Authorization guard validates session credentials prior to route execution.\n"
                "The guard denies unauthorized access and records audit logs.\n"
            ),
            "unrelated_notes.md": (
                "CSS palette and typography notes for landing page.\n"
                "Icon sizing and spacing guidance for components.\n"
            ),
        }

        for name, content in sample_files.items():
            (root / name).write_text(content, encoding="utf-8")

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
            _query = "How does middleware validate auth tokens before protected routes?"
            _pipeline_kwargs = {
                "conn": conn,
                "db_path": db_path,
                "query": _query,
                "top_k": 5,
                "fts_k": 8,
                "vector_k": 8,
                "min_fts_hits": DEFAULT_MIN_FTS_HITS,
                "hybrid_mode": "always",
                "retrieval_mode": "compact",
                "embedding_backend": "hash",
                "embedding_model": None,
                "session_id": "self-test",
                "query_cache_ttl_seconds": DEFAULT_QUERY_CACHE_TTL_SECONDS,
                "dimensions": DEFAULT_DIMENSIONS,
                "word_budget": DEFAULT_WORD_BUDGET,
                "relevance_floor": float(DEFAULT_RELEVANCE_FLOOR),
            }
            result = run_retrieval_pipeline(**_pipeline_kwargs)
            result_repeat = run_retrieval_pipeline(**_pipeline_kwargs)
        finally:
            conn.close()

    # Convert to dict for easier access in checks
    result_dict = result.model_dump()
    result_repeat_dict = result_repeat.model_dump()

    vector_retrieval_path = str(result_dict["retrieval"].get("vector_retrieval_path", ""))
    vector_hits = int(result_dict["retrieval"]["vector_hits"])
    vector_hits_expected = vector_retrieval_path not in {
        "disabled",
        "disabled_small_codebase",
        "skipped_hash_backend",
    }

    checks = {
        "bm25_enabled": bool(result_dict["retrieval"]["bm25_enabled"]),
        "fts_hits_gt_zero": int(result_dict["retrieval"]["fts_hits"]) > 0,
        "vector_hits_valid_for_mode": (vector_hits > 0) if vector_hits_expected else True,
        "selected_chunks_le_top_k": int(result_dict["selected_chunks"]) <= int(_pipeline_kwargs["top_k"]),
        "compressed_le_selected_tokens": (
            int(result_dict["token_metrics"]["compressed_tokens"])
            <= int(result_dict["token_metrics"]["selected_chunk_tokens"])
        ),
        "cache_hit_on_repeat": bool(result_repeat_dict.get("cache", {}).get("hit", False)),
        "session_memory_present": bool(
            result_repeat_dict.get("session_memory", {}).get("recent_queries")
        ),
    }

    ok = all(checks.values())
    summary = {
        "ok": ok,
        "checks": checks,
        "retrieval": result_dict["retrieval"],
        "token_metrics": result_dict["token_metrics"],
        "repeat_query_cache": result_repeat_dict.get("cache", {}),
    }
    typer.echo(json.dumps(summary, indent=2))
    if not ok:
        raise typer.Exit(1)


@app.command()
def benchmark(
    inputs: Annotated[
        list[str], typer.Option("--inputs", help="Files or directories to benchmark.")
    ],
    db: DbOpt = str(DEFAULT_DB_PATH),
    chunk_size: ChunkSizeOpt = _RUNTIME.chunk_size_words,
    overlap: OverlapOpt = _RUNTIME.chunk_overlap_words,
    embedding_backend: EmbBackendOpt = DEFAULT_EMBEDDING_BACKEND,
    embedding_model: EmbModelOpt = DEFAULT_EMBEDDING_MODEL,
    dimensions: DimensionsOpt = DEFAULT_DIMENSIONS,
    top_k: TopKOpt = _RUNTIME.default_top_k,
    fts_k: FtsKOpt = DEFAULT_FTS_K,
    vector_k: VectorKOpt = DEFAULT_VECTOR_K,
    hybrid_mode: HybridModeOpt = DEFAULT_HYBRID_MODE,
    word_budget: WordBudgetOpt = _RUNTIME.compression_word_budget,
    relevance_floor: RelevanceFloorOpt = _RUNTIME.relevance_floor,
) -> None:
    """Run comprehensive benchmarks and generate a metrics report."""
    import time

    db_path = Path(db).expanduser().resolve()

    files = collect_input_files(inputs)
    if not files:
        err.print("[red]No input files for benchmarking. Provide --inputs.[/red]")
        raise typer.Exit(1)

    raw_tokens_total = 0
    raw_chars_total = 0
    for file_path in files:
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="replace")
            tokens = estimate_tokens(content)
            raw_tokens_total += tokens
            raw_chars_total += len(content)
        except Exception:
            continue

    conn = connect_db(db_path)
    backend, model = resolve_embedding_backend(
        requested_backend=embedding_backend, requested_model=embedding_model
    )
    chunk_size_words, overlap_words = normalize_chunking(chunk_size, overlap)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=err) as progress:
        task = progress.add_task(f"Indexing {len(files)} file(s) for benchmark…")
        index_start = time.perf_counter()
        stats = index_corpus(
            conn=conn,
            file_paths=files,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            dimensions=dimensions,
            embedding_backend=backend,
            embedding_model=model,
        )
        index_latency_ms = (time.perf_counter() - index_start) * 1000
        progress.update(task, description="Building ANN index…")
        inferred_dims = infer_embedding_dimensions(
            conn=conn, backend=backend, model_name=model, fallback_dimensions=dimensions
        )
        build_hnsw_index(
            conn=conn, db_path=db_path, backend=backend, dimensions=inferred_dims, model_name=model
        )

    test_queries = [
        "How does authentication work?",
        "What functions handle database connections?",
        "Error handling patterns in this codebase",
        "Main entry point and initialization",
        "API endpoint implementations",
    ]

    query_metrics: list[dict] = []
    compressed_tokens_total = 0

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=err) as progress:
        task = progress.add_task(f"Running {len(test_queries)} benchmark queries…")
        for q in test_queries:
            q_start = time.perf_counter()
            result = run_retrieval_pipeline(
                conn=conn,
                db_path=db_path,
                query=q,
                top_k=top_k,
                fts_k=fts_k,
                vector_k=vector_k,
                min_fts_hits=DEFAULT_MIN_FTS_HITS,
                hybrid_mode=hybrid_mode,
                retrieval_mode="compact",
                embedding_backend=backend,
                embedding_model=model,
                session_id="benchmark",
                query_cache_ttl_seconds=0,
                dimensions=dimensions,
                word_budget=word_budget,
                relevance_floor=relevance_floor,
            )
            q_latency_ms = (time.perf_counter() - q_start) * 1000
            compressed_tokens = result.token_metrics.compressed_tokens
            selected_tokens = result.token_metrics.selected_chunk_tokens
            compressed_tokens_total += compressed_tokens
            query_metrics.append(
                {
                    "query": q,
                    "latency_ms": round(q_latency_ms, 2),
                    "fts_hits": result.retrieval.fts_hits,
                    "vector_hits": result.retrieval.vector_hits,
                    "selected_chunks": result.selected_chunks,
                    "selected_tokens": selected_tokens,
                    "compressed_tokens": compressed_tokens,
                    "compression_ratio": round(selected_tokens / max(1, compressed_tokens), 2),
                }
            )

    conn.close()

    avg_query_latency = sum(q["latency_ms"] for q in query_metrics) / len(query_metrics)
    avg_compression_ratio = sum(q["compression_ratio"] for q in query_metrics) / len(query_metrics)
    hypothetical_saved = raw_tokens_total - (compressed_tokens_total / len(test_queries))
    savings_pct = (hypothetical_saved / raw_tokens_total * 100) if raw_tokens_total > 0 else 0
    cost_per_1k = 0.01
    raw_cost = (raw_tokens_total / 1000) * cost_per_1k
    compressed_cost = (compressed_tokens_total / len(test_queries) / 1000) * cost_per_1k
    cost_savings = raw_cost - compressed_cost

    report = {
        "benchmark_summary": {
            "files_indexed": len(files),
            "chunks_created": stats["chunks_indexed"],
            "raw_input_tokens": raw_tokens_total,
            "raw_input_chars": raw_chars_total,
            "embedding_backend": backend,
            "embedding_model": model,
        },
        "latency_metrics": {
            "index_time_ms": round(index_latency_ms, 2),
            "avg_query_time_ms": round(avg_query_latency, 2),
            "min_query_time_ms": round(min(q["latency_ms"] for q in query_metrics), 2),
            "max_query_time_ms": round(max(q["latency_ms"] for q in query_metrics), 2),
        },
        "compression_metrics": {
            "avg_compression_ratio": round(avg_compression_ratio, 2),
            "avg_compressed_tokens_per_query": round(
                compressed_tokens_total / len(test_queries), 0
            ),
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

    typer.echo(json.dumps(report, indent=2))

    err.print("\n[bold]TOKEN REDUCER BENCHMARK[/bold]")
    err.print(f"  Files indexed:          {len(files)}")
    err.print(f"  Raw input tokens:       {raw_tokens_total:,}")
    err.print(f"  Avg compressed tokens:  {compressed_tokens_total // len(test_queries):,}")
    err.print(f"  Token savings:          [green]{savings_pct:.1f}%[/green]")
    err.print(f"  Avg query latency:      {avg_query_latency:.1f} ms")
    err.print(f"  Cost savings / 100q:    [green]${cost_savings * 100:.2f}[/green]")


@app.command("benchmark-full")
def benchmark_full(
    inputs: Annotated[
        list[str], typer.Option("--inputs", help="Files or directories to benchmark.")
    ],
    db: DbOpt = str(DEFAULT_DB_PATH),
    chunk_size: ChunkSizeOpt = _RUNTIME.chunk_size_words,
    overlap: OverlapOpt = _RUNTIME.chunk_overlap_words,
    embedding_backend: EmbBackendOpt = DEFAULT_EMBEDDING_BACKEND,
    embedding_model: EmbModelOpt = DEFAULT_EMBEDDING_MODEL,
    dimensions: DimensionsOpt = DEFAULT_DIMENSIONS,
    top_k: TopKOpt = _RUNTIME.default_top_k,
    fts_k: FtsKOpt = DEFAULT_FTS_K,
    vector_k: VectorKOpt = DEFAULT_VECTOR_K,
    hybrid_mode: HybridModeOpt = DEFAULT_HYBRID_MODE,
    word_budget: WordBudgetOpt = _RUNTIME.compression_word_budget,
    relevance_floor: RelevanceFloorOpt = _RUNTIME.relevance_floor,
    iterations: Annotated[int, typer.Option("--iterations", help="Runs per query for latency stats.")] = 3,
    output_format: Annotated[str, typer.Option("--format", help="json | markdown")] = "json",
    skip_accuracy: Annotated[bool, typer.Option("--skip-accuracy", help="Skip accuracy metrics.")] = False,
    skip_stress: Annotated[bool, typer.Option("--skip-stress", help="Skip stress test queries.")] = False,
) -> None:
    """Run comprehensive benchmarks with accuracy metrics, latency percentiles, and cost analysis.

    This command provides scientifically rigorous benchmarks including:
    - Before/after token counts with baseline comparison
    - Latency impact (p50/p95/p99)
    - Accuracy metrics (Precision@K, Recall@K, MRR, NDCG)
    - Real dollar savings projections

    Examples:
        # Run full benchmark suite
        token-reducer benchmark-full --inputs ./src

        # Generate markdown report
        token-reducer benchmark-full --inputs ./src --format markdown

        # Quick benchmark (skip accuracy tests)
        token-reducer benchmark-full --inputs ./src --skip-accuracy
    """
    from .benchmark import (
        BenchmarkConfig,
        format_benchmark_json,
        format_benchmark_markdown,
        run_comprehensive_benchmark,
    )

    db_path = Path(db).expanduser().resolve()
    input_paths = [Path(p).expanduser().resolve() for p in inputs]

    valid_inputs = [p for p in input_paths if p.exists()]
    if not valid_inputs:
        err.print("[red]No valid input paths found.[/red]")
        raise typer.Exit(1)

    config = BenchmarkConfig(
        inputs=valid_inputs,
        db_path=db_path,
        top_k=top_k,
        fts_k=fts_k,
        vector_k=vector_k,
        hybrid_mode=hybrid_mode,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        dimensions=dimensions,
        chunk_size=chunk_size,
        overlap=overlap,
        word_budget=word_budget,
        relevance_floor=relevance_floor,
        num_iterations=iterations,
    )

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=err) as progress:
        progress.add_task("Running comprehensive benchmark suite…")

        try:
            report = run_comprehensive_benchmark(
                config=config,
                include_accuracy=not skip_accuracy,
                include_stress=not skip_stress,
            )
        except ValueError as e:
            err.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    if output_format == "markdown":
        typer.echo(format_benchmark_markdown(report))
    else:
        typer.echo(format_benchmark_json(report))

    # Print summary to stderr
    err.print("\n[bold]COMPREHENSIVE BENCHMARK RESULTS[/bold]")
    err.print(f"  Files indexed:          {report['benchmark_summary']['files_indexed']}")
    err.print(f"  Baseline tokens:        {report['baseline_comparison']['baseline_tokens']:,}")
    err.print(f"  Compressed tokens/query: ~{report['baseline_comparison']['avg_compressed_tokens_per_query']:,.0f}")
    err.print(f"  Token reduction:        [green]{report['baseline_comparison']['reduction_vs_baseline_pct']}%[/green]")
    err.print(f"  p50 latency:            {report['latency_metrics']['p50_query_ms']} ms")
    err.print(f"  p95 latency:            {report['latency_metrics']['p95_query_ms']} ms")
    if not skip_accuracy:
        err.print(f"  Precision@K:            {report['accuracy_metrics']['precision_at_k']}")
        err.print(f"  Recall@K:               {report['accuracy_metrics']['recall_at_k']}")
    err.print(f"  Cost savings/100q:      [green]${report['cost_analysis']['savings_per_100_queries_usd']:.2f}[/green]")


# ---------------------------------------------------------------------------
# Context Lifecycle Management Commands
# ---------------------------------------------------------------------------


@app.command()
def sync(
    inputs: Annotated[list[str], typer.Option("--inputs", help="Files or directories to sync.")],
    db: DbOpt = str(DEFAULT_DB_PATH),
    chunk_size: ChunkSizeOpt = _RUNTIME.chunk_size_words,
    overlap: OverlapOpt = _RUNTIME.chunk_overlap_words,
    embedding_backend: EmbBackendOpt = DEFAULT_EMBEDDING_BACKEND,
    embedding_model: EmbModelOpt = DEFAULT_EMBEDDING_MODEL,
    dimensions: DimensionsOpt = DEFAULT_DIMENSIONS,
    output_json: JsonOpt = False,
) -> None:
    """Incrementally sync the index with the filesystem.

    Only re-indexes files that have been added or modified since last sync.
    Removes index entries for files that no longer exist.
    """
    import os

    db_path = Path(db).expanduser().resolve()
    conn = connect_db(db_path)
    try:
        files = collect_input_files(inputs)
        if not files:
            err.print("[yellow]No indexable files found in inputs.[/yellow]")
            raise typer.Exit(1)

        backend, model = resolve_embedding_backend(
            requested_backend=embedding_backend,
            requested_model=embedding_model,
        )
        chunk_size_words, overlap_words = normalize_chunking(chunk_size, overlap)

        # Detect changes
        new_files, modified_files, deleted_sources = detect_file_changes(conn, files)

        stats = {
            "new_files": len(new_files),
            "modified_files": len(modified_files),
            "deleted_files": len(deleted_sources),
            "unchanged_files": len(files) - len(new_files) - len(modified_files),
            "chunks_indexed": 0,
        }

        # If no changes detected
        if not new_files and not modified_files and not deleted_sources:
            if output_json:
                typer.echo(json.dumps({"status": "up_to_date", **stats}, indent=2))
            else:
                err.print("[green]✓ Context is up to date. No sync needed.[/green]")
            return

        # Remove deleted files from index
        if deleted_sources:
            removed = remove_documents_by_source(conn, deleted_sources)
            if not output_json:
                err.print(f"[dim]Removed {removed} deleted file(s) from index.[/dim]")

        # Index new and modified files
        files_to_index = new_files + modified_files
        if files_to_index:
            with Progress(
                SpinnerColumn(), TextColumn("{task.description}"), console=err
            ) as progress:
                progress.add_task(f"Syncing {len(files_to_index)} file(s)…")

                for path in files_to_index:
                    raw = path.read_text(encoding="utf-8", errors="replace")
                    cleaned = clean_text(raw)
                    if not cleaned:
                        continue

                    chunk_count = upsert_document(
                        conn=conn,
                        source=str(path),
                        raw_text=raw,
                        cleaned_text=cleaned,
                        chunk_size_words=chunk_size_words,
                        overlap_words=overlap_words,
                        dimensions=dimensions,
                        embedding_backend=backend,
                        embedding_model=model,
                    )
                    stats["chunks_indexed"] += chunk_count

                    # Update mtime for change tracking
                    mtime = os.path.getmtime(path)
                    update_document_mtime(conn, str(path), mtime)

                conn.commit()

            # Rebuild ANN index if we have enough chunks
            adaptive_tier = infer_retrieval_tier(conn)
            if adaptive_tier == "full_hybrid":
                inferred_dims = infer_embedding_dimensions(
                    conn=conn, backend=backend, model_name=model, fallback_dimensions=dimensions
                )
                build_hnsw_index(
                    conn=conn,
                    db_path=db_path,
                    backend=backend,
                    dimensions=inferred_dims,
                    model_name=model,
                )

        if output_json:
            typer.echo(json.dumps({"status": "synced", **stats}, indent=2))
        else:
            err.print("[green]✓ Sync complete:[/green]")
            err.print(f"    New files indexed:      {stats['new_files']}")
            err.print(f"    Modified files updated: {stats['modified_files']}")
            err.print(f"    Deleted files removed:  {stats['deleted_files']}")
            err.print(f"    Total chunks indexed:   {stats['chunks_indexed']}")

    finally:
        conn.close()


@app.command()
def gc(
    db: DbOpt = str(DEFAULT_DB_PATH),
    vacuum: Annotated[bool, typer.Option("--vacuum", help="Run VACUUM after cleanup.")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be cleaned without deleting.")
    ] = False,
    output_json: JsonOpt = False,
) -> None:
    """Garbage collect stale tokens and orphaned data.

    Cleans up:
    - Orphaned chunks (no parent document)
    - Orphaned embeddings (no parent chunk)
    - Expired query cache entries
    - Orphaned symbols and dependencies
    """
    db_path = Path(db).expanduser().resolve()
    if not db_path.exists():
        err.print(f"[red]Database not found: {db_path}[/red]")
        raise typer.Exit(1)

    conn = connect_db(db_path)
    try:
        size_before = get_database_size(db_path)

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=err) as progress:
            task_desc = "Analyzing database…" if dry_run else "Running garbage collection…"
            progress.add_task(task_desc)
            gc_stats = garbage_collect(conn, dry_run=dry_run)

        if vacuum and not dry_run:
            with Progress(
                SpinnerColumn(), TextColumn("{task.description}"), console=err
            ) as progress:
                progress.add_task("Running VACUUM…")
                vacuum_database(conn)

        size_after = get_database_size(db_path)
        bytes_reclaimed = size_before - size_after

        result = {
            "dry_run": dry_run,
            "orphaned_chunks_removed": gc_stats["orphaned_chunks"],
            "orphaned_embeddings_removed": gc_stats["orphaned_embeddings"],
            "expired_cache_entries_removed": gc_stats["expired_cache_entries"],
            "orphaned_symbols_removed": gc_stats["orphaned_symbols"],
            "orphaned_dependencies_removed": gc_stats["orphaned_dependencies"],
            "vacuum_run": vacuum and not dry_run,
            "bytes_reclaimed": bytes_reclaimed if not dry_run else 0,
        }

        total_cleaned = sum(
            [
                gc_stats["orphaned_chunks"],
                gc_stats["orphaned_embeddings"],
                gc_stats["expired_cache_entries"],
                gc_stats["orphaned_symbols"],
                gc_stats["orphaned_dependencies"],
            ]
        )

        if output_json:
            typer.echo(json.dumps(result, indent=2))
        else:
            action = "Would clean" if dry_run else "Cleaned"
            if total_cleaned == 0:
                err.print("[green]✓ Database clean. No garbage collection needed.[/green]")
            else:
                err.print(f"[green]✓ {action}:[/green]")
                if gc_stats["orphaned_chunks"]:
                    err.print(f"    Orphaned chunks:      {gc_stats['orphaned_chunks']}")
                if gc_stats["orphaned_embeddings"]:
                    err.print(f"    Orphaned embeddings:  {gc_stats['orphaned_embeddings']}")
                if gc_stats["expired_cache_entries"]:
                    err.print(f"    Expired cache:        {gc_stats['expired_cache_entries']}")
                if gc_stats["orphaned_symbols"]:
                    err.print(f"    Orphaned symbols:     {gc_stats['orphaned_symbols']}")
                if gc_stats["orphaned_dependencies"]:
                    err.print(f"    Orphaned deps:        {gc_stats['orphaned_dependencies']}")

            if vacuum and not dry_run and bytes_reclaimed > 0:
                err.print(f"    Bytes reclaimed:      {bytes_reclaimed:,}")

    finally:
        conn.close()


@app.command()
def stats(
    db: DbOpt = str(DEFAULT_DB_PATH),
    output_json: JsonOpt = False,
) -> None:
    """Show index statistics and health information."""
    db_path = Path(db).expanduser().resolve()
    if not db_path.exists():
        err.print(f"[red]Database not found: {db_path}[/red]")
        raise typer.Exit(1)

    conn = connect_db(db_path)
    try:
        index_stats = get_index_stats(conn)
        db_size = get_database_size(db_path)
        tier = infer_retrieval_tier(conn)

        result = {
            "database_path": str(db_path),
            "database_size_bytes": db_size,
            "retrieval_tier": tier,
            **index_stats,
        }

        if output_json:
            typer.echo(json.dumps(result, indent=2))
        else:
            err.print("[bold]TOKEN REDUCER INDEX STATS[/bold]")
            err.print(f"  Database:        {db_path}")
            err.print(f"  Size:            {db_size / 1024:.1f} KB")
            err.print(f"  Retrieval tier:  {tier}")
            err.print(f"  Documents:       {index_stats['documents']}")
            err.print(f"  Chunks:          {index_stats['chunks']}")
            err.print(f"  Embeddings:      {index_stats['embeddings']}")
            err.print(f"  Symbols:         {index_stats['symbols']}")
            err.print(f"  Dependencies:    {index_stats['file_dependencies']}")
            err.print(f"  Cache entries:   {index_stats['query_cache_entries']}")
            if index_stats["oldest_document"]:
                err.print(f"  Oldest doc:      {index_stats['oldest_document']}")
            if index_stats["newest_document"]:
                err.print(f"  Newest doc:      {index_stats['newest_document']}")
            if index_stats["embedding_backends"]:
                backends = ", ".join(
                    f"{k}: {v}" for k, v in index_stats["embedding_backends"].items()
                )
                err.print(f"  Backends:        {backends}")

    finally:
        conn.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
