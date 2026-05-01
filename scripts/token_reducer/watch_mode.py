"""Debounced filesystem poll → incremental re-index (no extra deps)."""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .chunker import collect_input_files, normalize_chunking
from .db import connect_db
from .incremental_index import update_index_incremental


def run_watch(
    inputs: list[str],
    db_path: Path,
    chunk_size_words: int,
    overlap_words: int,
    dimensions: int,
    embedding_backend: str,
    embedding_model: str | None,
    debounce_s: float = 1.5,
    poll_interval_s: float = 2.0,
    on_tick: Callable[[dict[str, int]], None] | None = None,
) -> None:
    """Poll filesystem fingerprints; debounce incremental index updates."""
    last_fp = ""
    timer: threading.Timer | None = None
    lock = threading.Lock()

    def fingerprint(paths: list[Path]) -> str:
        parts: list[str] = []
        for p in sorted(set(paths)):
            try:
                if p.is_file():
                    st = p.stat()
                    parts.append(f"{p.resolve()}:{st.st_mtime_ns}:{st.st_size}")
            except OSError:
                continue
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    def do_index() -> None:
        paths = collect_input_files(inputs)
        conn = connect_db(db_path)
        try:
            cs, ov = normalize_chunking(chunk_size_words, overlap_words)
            stats = update_index_incremental(
                conn,
                paths,
                chunk_size_words=cs,
                overlap_words=ov,
                dimensions=dimensions,
                embedding_backend=embedding_backend,
                embedding_model=embedding_model,
            )
            if on_tick:
                on_tick(stats)
        finally:
            conn.close()

    def schedule() -> None:
        nonlocal timer
        with lock:
            if timer is not None:
                timer.cancel()
            timer = threading.Timer(debounce_s, do_index)
            timer.daemon = True
            timer.start()

    while True:
        time.sleep(poll_interval_s)
        paths = collect_input_files(inputs)
        fp = fingerprint(paths)
        if fp != last_fp:
            last_fp = fp
            schedule()
