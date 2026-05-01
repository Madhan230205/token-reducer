---
description: Build a compact context packet with preprocess, BM25-ranked FTS retrieval, adaptive vector fallback, reranked top-K chunks (default 3 from settings), relevance-floor filtering, and compression.
argument-hint: query [--inputs path1,path2] [--top-k 3-5] [--relevance-floor 0.15-0.25] [--hybrid-mode fallback|always] [--session-id id] [--json]
allowed-tools: [Read, Glob, Grep, Bash, Task]
---

# token-reducer

Run the full token-slim pipeline and return a context packet suitable for Claude reasoning.

## Input

User arguments:

`$ARGUMENTS`

## Execution plan

1. If no explicit input paths are provided, default to current workspace root.
2. Run pipeline script in `run` mode.
3. Enforce FTS-first retrieval with BM25 scoring.
4. Run vector retrieval adaptively (fallback default), merge results, and keep top **K** reranked chunks (plugin default **3** via `settings.json` → `defaultTopK`).
5. Apply **`relevanceFloor`** so low-scoring chunks are not summarized.
6. Compress before response handoff.

## Primary command

`python "${CLAUDE_PLUGIN_ROOT}/scripts/context_pipeline.py" run --inputs . --query "$ARGUMENTS" --hybrid-mode fallback --session-id default --top-k 3`

For stricter cost control, keep `--top-k 3` and a **specific** query (e.g. “Find the JWT validation function”) so irrelevant files score out before compression.

## Notes

- Prefer local pipeline to stay free-cost.
- ANN acceleration is used automatically when `hnswlib` + `numpy` are available.
- Use Context7 only when external library docs are needed.
- Always return citations and estimated savings.
- Large raw pasted query blobs are guarded; put logs/files in `--inputs`.
- Default chunk size, word budget, `top-k`, and relevance floor are loaded from the plugin **`settings.json`** when `CLAUDE_PLUGIN_ROOT` is set (repo `settings.json` when running from a checkout).
