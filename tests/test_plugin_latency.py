from __future__ import annotations

import time

from token_reducer.intent import analyze_query_intent


def test_intent_latency_ms() -> None:
    t0 = time.perf_counter()
    for _ in range(200):
        analyze_query_intent("How do we refactor the database layer for tests?")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 100.0
