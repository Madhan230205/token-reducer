"""MCP server: exposes the local retrieval engine as tools any MCP client can call.

Run: ``python scripts/token_reducer_mcp.py`` (stdio). Workspace root defaults to cwd,
override with ``TOKEN_REDUCER_WORKSPACE``. Index lives outside the repo under
``~/.cache/token-reducer/workspaces/<hash>/index.db`` (override: ``TOKEN_REDUCER_DB``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .ann import build_hnsw_index
from .chunker import collect_input_files, is_code_file, normalize_chunking, read_text_file, tokenize
from .config import (
    DEFAULT_DIMENSIONS,
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FTS_K,
    DEFAULT_VECTOR_K,
)
from .db import (
    connect_db,
    get_index_stats,
    load_session_memory,
    save_session_memory,
    session_memory_path,
)
from .embeddings import embed_texts_onnx, resolve_embedding_backend
from .incremental_index import update_index_incremental
from .models import Candidate
from .pipeline import run_retrieval_pipeline
from .plugin_settings import get_runtime_defaults
from .retriever import fts_retrieve, rerank_candidates, vector_retrieve
from .structure import infer_chunk_meta

SYNC_INTERVAL_SECONDS = 20
PREVIEW_CHARS = 420
MAX_TOP_K = 25
EMBED_BATCH = 32
MIN_RESULT_WORDS = 8
CODE_FILE_BOOST = 1.25
SYMBOL_MATCH_BOOST = 1.5
TEST_FILE_PENALTY = 0.8


def workspace_root() -> Path:
    raw = os.environ.get("TOKEN_REDUCER_WORKSPACE", "").strip()
    return Path(raw).expanduser().resolve() if raw else Path.cwd().resolve()


def db_path_for(root: Path) -> Path:
    raw = os.environ.get("TOKEN_REDUCER_DB", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    digest = hashlib.sha1(str(root).lower().encode("utf-8")).hexdigest()[:16]
    return Path.home() / ".cache" / "token-reducer" / "workspaces" / digest / "index.db"


def workspace_files(root: Path) -> list[Path]:
    """Prefer ``git ls-files`` (respects .gitignore); fall back to the directory walk."""
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return collect_input_files([str(root)])
    names = [n for n in out.decode("utf-8", errors="replace").split("\0") if n]
    return collect_input_files([str(root / n) for n in names])


def rel(root: Path, source: str) -> str:
    try:
        return Path(source).resolve().relative_to(root).as_posix()
    except ValueError:
        return source


class Engine:
    """One workspace index; lazily synced, throttled."""

    def __init__(
        self,
        root: Path | None = None,
        db_path: Path | None = None,
        *,
        background_embeddings: bool = True,
    ) -> None:
        # Headless LSP lookups spawn language servers with blocking, untimed reads —
        # unacceptable inside a tool call. Opt back in with TOKEN_REDUCER_DISABLE_LSP=0.
        os.environ.setdefault("TOKEN_REDUCER_DISABLE_LSP", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.root = root or workspace_root()
        self.db_path = db_path or db_path_for(self.root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = connect_db(self.db_path, check_same_thread=False)
        self.rt = get_runtime_defaults()
        self.backend, self.model = resolve_embedding_backend(
            requested_backend=os.environ.get(
                "TOKEN_REDUCER_EMBEDDING_BACKEND", DEFAULT_EMBEDDING_BACKEND
            ),
            requested_model=DEFAULT_EMBEDDING_MODEL,
        )
        self._lock = threading.RLock()
        self._embed_lock = threading.Lock()  # one ONNX/tokenizer caller at a time
        self._last_sync = 0.0
        self._last_sync_stats: dict[str, int] = {}
        self._upgrade_thread: threading.Thread | None = None
        self.background_embeddings = background_embeddings and self.backend == "onnx"

    def sync(self, force: bool = False) -> dict[str, int]:
        """Index new/changed files. FTS is ready immediately; dense vectors fill in behind."""
        with self._lock:
            now = time.monotonic()
            if not force and now - self._last_sync < SYNC_INTERVAL_SECONDS:
                return self._last_sync_stats
            files = workspace_files(self.root)
            chunk, overlap = normalize_chunking(
                self.rt.chunk_size_words, self.rt.chunk_overlap_words
            )
            fast = self.background_embeddings
            stats = update_index_incremental(
                self.conn,
                files,
                chunk_size_words=chunk,
                overlap_words=overlap,
                dimensions=DEFAULT_DIMENSIONS,
                embedding_backend="hash" if fast else self.backend,
                embedding_model=None if fast else self.model,
            )
            stats["files_seen"] = len(files)
            self._last_sync = time.monotonic()
            self._last_sync_stats = stats
        if self.background_embeddings and stats.get("chunks", 0):
            self._start_upgrade()
        return stats

    def _start_upgrade(self) -> None:
        if self._upgrade_thread and self._upgrade_thread.is_alive():
            return
        self._upgrade_thread = threading.Thread(target=self.upgrade_embeddings, daemon=True)
        self._upgrade_thread.start()

    def _forget_session(self, sid: str) -> None:
        with contextlib.suppress(Exception):
            path = session_memory_path(self.db_path)
            memory = load_session_memory(path)
            if memory.get("sessions", {}).pop(sid, None) is not None:
                save_session_memory(path, memory)

    def _dense_ready(self) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM chunk_embeddings WHERE backend = 'hash'"
            ).fetchone()
        return not row or int(row[0]) == 0

    def pending_embeddings(self) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM chunk_embeddings WHERE backend = 'hash'"
            ).fetchone()
        return int(row[0]) if row else 0

    def upgrade_embeddings(self, batch: int = EMBED_BATCH) -> int:
        """Replace hash embeddings with dense ONNX vectors, in batches. Returns rows upgraded."""
        if self.backend != "onnx" or not self.model:
            return 0
        done = 0
        dims = DEFAULT_DIMENSIONS
        while True:
            with self._lock:
                rows = self.conn.execute(
                    "SELECT ce.chunk_id, c.text FROM chunk_embeddings ce JOIN chunks c ON c.id = ce.chunk_id "
                    "WHERE ce.backend = 'hash' ORDER BY ce.chunk_id LIMIT ?",
                    (batch,),
                ).fetchall()
            if not rows:
                break
            try:
                with self._embed_lock:
                    vecs = embed_texts_onnx([str(r["text"]) for r in rows], self.model)
            except Exception as exc:
                print(f"[token-reducer] dense embedding upgrade stopped: {exc}", file=sys.stderr)
                return done
            with self._lock:
                self.conn.executemany(
                    "UPDATE chunk_embeddings SET embedding_json = ?, backend = 'onnx', dimensions = ?, model_name = ? "
                    "WHERE chunk_id = ?",
                    [
                        (
                            json.dumps(v, separators=(",", ":")),
                            len(v),
                            self.model,
                            int(r["chunk_id"]),
                        )
                        for r, v in zip(rows, vecs, strict=True)
                    ],
                )
                self.conn.commit()
            done += len(rows)
            dims = len(vecs[0]) if vecs else dims
        if done:
            with self._lock, contextlib.suppress(Exception):
                build_hnsw_index(
                    conn=self.conn,
                    db_path=self.db_path,
                    backend="onnx",
                    dimensions=dims,
                    model_name=self.model,
                )
        return done

    def search(self, query: str, top_k: int) -> list[dict[str, object]]:
        self.sync()
        top_k = max(1, min(int(top_k), MAX_TOP_K))
        pool = max(30, top_k * 5)
        with self._lock:
            fts = fts_retrieve(self.conn, query, limit=pool)
            vec: list = []
            # Partial dense coverage poisons rank fusion (a vector rank-1 among 30 embedded
            # chunks is noise), so vectors join only once the background upgrade is done.
            if self.backend != "hash" and self._dense_ready():
                with self._embed_lock, contextlib.suppress(Exception):
                    vec, _, _, _ = vector_retrieve(
                        conn=self.conn,
                        db_path=self.db_path,
                        query=query,
                        limit=pool,
                        dimensions=DEFAULT_DIMENSIONS,
                        embedding_backend=self.backend,
                        embedding_model=self.model,
                    )
        ranked, _ = rerank_candidates(query, fts, vec, top_k=pool)
        q_terms = {t for t in tokenize(query) if len(t) >= 3}
        wants_tests = bool(q_terms & {"test", "tests", "spec", "fixture"})
        scored: list[tuple[float, Candidate, dict[str, str | int]]] = []
        for c in ranked:
            if len(c.text.split()) < MIN_RESULT_WORDS and len(ranked) > top_k:
                continue
            meta = infer_chunk_meta(c.text, c.source)
            score = float(c.final_score)
            if is_code_file(c.source):
                score *= CODE_FILE_BOOST
            if not wants_tests and _looks_like_test(c.source):
                score *= TEST_FILE_PENALTY
            sym = str(meta.get("symbol_name", "")).lower()
            if sym and q_terms & set(tokenize(sym)):
                score *= SYMBOL_MATCH_BOOST
            scored.append((score, c, meta))
        scored.sort(key=lambda t: t[0], reverse=True)
        out: list[dict[str, object]] = []
        file_cache: dict[str, str | None] = {}
        for score, c, meta in scored[:top_k]:
            c.final_score = score
            start, end = _locate_lines(c.source, c.text, file_cache)
            out.append(
                {
                    "file": rel(self.root, c.source),
                    "symbol": str(meta.get("symbol_name", "")),
                    "kind": str(meta.get("chunk_type", "text")),
                    "start_line": start,
                    "end_line": end,
                    "score": round(float(c.final_score), 3),
                    "preview": c.text[:PREVIEW_CHARS].rstrip(),
                }
            )
        return out

    def context(self, query: str, top_k: int) -> dict[str, object]:
        self.sync()
        dense = self._dense_ready()
        # Fresh session per call: the pipeline's session memory omits chunks it already served
        # to a session, which would make a repeated question return "nothing matched".
        sid = f"mcp-{uuid.uuid4().hex[:8]}"
        with self._lock, self._embed_lock:
            pkt = run_retrieval_pipeline(
                conn=self.conn,
                db_path=self.db_path,
                query=query,
                top_k=max(1, min(int(top_k), 8)),
                fts_k=DEFAULT_FTS_K,
                vector_k=DEFAULT_VECTOR_K,
                min_fts_hits=2,
                hybrid_mode="fallback",
                retrieval_mode="compact",
                embedding_backend=self.backend if dense else "hash",
                embedding_model=self.model if dense else None,
                session_id=sid,
                query_cache_ttl_seconds=300,
                dimensions=DEFAULT_DIMENSIONS,
                word_budget=self.rt.compression_word_budget,
                relevance_floor=self.rt.relevance_floor,
                workspace_root=self.root,
            )
            self._forget_session(sid)
        ctx = dict(pkt.claude_context or {})
        for item in ctx.get("code_context", []) or []:
            if isinstance(item, dict) and isinstance(item.get("file"), str):
                item["file"] = rel(self.root, item["file"])
        ctx["relevant_files"] = [rel(self.root, f) for f in ctx.get("relevant_files", []) or []]
        for key in ("_routing_debug", "_context_adaptation", "decision_trace"):
            ctx.pop(key, None)
        ctx["tokens"] = {
            "selected_chunks": pkt.token_metrics.selected_chunk_tokens,
            "compressed": pkt.token_metrics.compressed_tokens,
        }
        return ctx

    def symbol(self, name: str, top_k: int) -> list[dict[str, str]]:
        self.sync()
        with self._lock:
            hits = fts_retrieve(self.conn, f'"{name}"', limit=max(1, min(int(top_k), MAX_TOP_K)))
        out = []
        for h in hits:
            idx = h.text.find(name)
            lo = max(0, idx - 160)
            out.append({"file": rel(self.root, h.source), "snippet": h.text[lo : lo + 360].strip()})
        return out

    def status(self) -> dict[str, object]:
        self.sync()
        with self._lock:
            stats = get_index_stats(self.conn)
        return {
            "workspace": str(self.root),
            "db": str(self.db_path),
            "embedding_backend": self.backend,
            "documents": stats["documents"],
            "chunks": stats["chunks"],
            "dense_embeddings_pending": self.pending_embeddings() if self.backend == "onnx" else 0,
            "last_sync": self._last_sync_stats,
        }

    def read_slice(self, path: str, start: int, end: int) -> dict[str, object]:
        p = (self.root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if self.root not in p.parents and p != self.root:
            return {"ok": False, "error": "path outside workspace"}
        raw = read_text_file(p)
        if raw is None:
            return {"ok": False, "error": "unreadable", "path": str(p)}
        lines = raw.splitlines()
        start = max(1, int(start))
        end = min(len(lines), int(end) if end > 0 else len(lines))
        body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        return {
            "ok": True,
            "file": rel(self.root, str(p)),
            "start": start,
            "end": end,
            "total_lines": len(lines),
            "content": body,
        }


def _locate_lines(source: str, chunk: str, cache: dict[str, str | None]) -> tuple[int, int]:
    """Real 1-based line span of ``chunk`` inside its file (0, 0 when it can't be found)."""
    if source not in cache:
        cache[source] = read_text_file(Path(source))
    body = cache[source]
    if not body:
        return 0, 0
    first = next((ln.strip() for ln in chunk.splitlines() if ln.strip()), "")
    if not first:
        return 0, 0
    lines = body.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip() == first:
            n = len(chunk.splitlines())
            return i + 1, min(len(lines), i + n)
    return 0, 0


def _looks_like_test(source: str) -> bool:
    parts = Path(source).as_posix().lower().split("/")
    name = parts[-1]
    return (
        any(p in {"test", "tests", "__tests__", "spec", "specs"} for p in parts[:-1])
        or name.startswith("test_")
        or name.endswith(("_test.py", ".test.ts", ".test.js", ".spec.ts", ".spec.js", "_test.go"))
    )


def _fmt_search(hits: list[dict[str, object]]) -> str:
    if not hits:
        return "No matches. Try different terms, or call read_file_slice on a likely file."
    parts = []
    for h in hits:
        head = f"{h['file']}"
        if h.get("start_line"):
            head += f":{h['start_line']}-{h['end_line']}"
        if h["symbol"]:
            head += f"  [{h['kind']} {h['symbol']}]"
        head += f"  score={h['score']}"
        parts.append(head + "\n" + str(h["preview"]))
    return "\n\n".join(parts)


def build_server(engine: Engine | None = None):  # type: ignore[no-untyped-def]
    from mcp.server.fastmcp import FastMCP

    eng = engine or Engine()
    mcp = FastMCP(
        "token-reducer",
        instructions=(
            "Local hybrid code search (BM25 + embeddings) over the current workspace. "
            "Call search_code BEFORE reading whole files or grepping broadly: it returns the few "
            "most relevant chunks, which usually answers 'where is X / how does Y work' in one call. "
            "Use get_context when you want a compressed brief for a task. Use read_file_slice to "
            "pull exact line ranges instead of entire files."
        ),
    )

    @mcp.tool()
    def search_code(query: str, top_k: int = 6) -> str:
        """Find the most relevant code chunks for a natural-language or keyword query.

        Returns file path, symbol, score and a short preview per hit. Cheap and fast;
        prefer this over Grep/Read when you don't yet know which file to look at.
        """
        return _fmt_search(eng.search(query, top_k))

    @mcp.tool()
    def get_context(task: str, top_k: int = 3) -> str:
        """Build a compact, token-budgeted context brief for a coding task.

        Runs the full pipeline (intent → hybrid retrieval → rerank → compression) and returns
        JSON with summary, relevant_files, relationships and trimmed code_context.
        """
        return json.dumps(eng.context(task, top_k), ensure_ascii=False, indent=1)

    @mcp.tool()
    def find_symbol(name: str, top_k: int = 10) -> str:
        """List files and snippets where an exact identifier (function, class, constant) appears."""
        hits = eng.symbol(name, top_k)
        if not hits:
            return f"No occurrences of '{name}'."
        return "\n\n".join(f"{h['file']}\n{h['snippet']}" for h in hits)

    @mcp.tool()
    def read_file_slice(path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a line range of a workspace file (1-based, inclusive; end_line 0 = EOF).

        Use after search_code points you at a file, instead of reading the whole file.
        """
        res = eng.read_slice(path, start_line, end_line)
        if not res.get("ok"):
            return f"error: {res.get('error')}"
        return f"{res['file']} lines {res['start']}-{res['end']} of {res['total_lines']}\n{res['content']}"

    @mcp.tool()
    def index_status() -> str:
        """Show workspace index stats (documents, chunks, embedding backend, last sync)."""
        return json.dumps(eng.status(), ensure_ascii=False, indent=1)

    @mcp.tool()
    def reindex() -> str:
        """Force an incremental re-sync of the index with the filesystem."""
        return json.dumps(eng.sync(force=True), ensure_ascii=False)

    return mcp


class _ProtectedStdout:
    """Keeps the real stdout buffer for the MCP transport while sending every
    Python-level ``print`` (ours or a dependency's) to stderr. A stray line on
    stdout would corrupt the JSON-RPC stream and hang the client."""

    def __init__(self, real: object) -> None:
        self.buffer = getattr(real, "buffer", None)
        self.encoding = "utf-8"

    def write(self, data: str) -> int:
        return sys.stderr.write(data)

    def flush(self) -> None:
        sys.stderr.flush()

    def fileno(self) -> int:
        return self.buffer.fileno()  # type: ignore[union-attr]

    def isatty(self) -> bool:
        return False


def main() -> int:
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout = _ProtectedStdout(sys.stdout)  # type: ignore[assignment]
    if os.environ.get("TOKEN_REDUCER_MCP_DEBUG_HANG"):
        import faulthandler

        faulthandler.dump_traceback_later(20, repeat=True, file=sys.stderr)
    build_server().run(transport="stdio")
    return 0
