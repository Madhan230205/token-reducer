---
description: Build a compact context packet with preprocess, BM25-ranked FTS retrieval, adaptive vector fallback, reranked top-K chunks (default 3 from settings), relevance-floor filtering, and compression.
argument-hint: query [--inputs path1,path2] [--top-k 3-5] [--relevance-floor 0.15-0.25] [--hybrid-mode fallback|always] [--session-id id] [--json]
allowed-tools: [Read, Glob, Grep, Bash, Task]
---

# token-reducer

Run the **full** pipeline in one shot: indexing (optional), hybrid retrieval, intent + execution routing, subagents, compression, session memory, and (when `CLAUDE_PLUGIN_ROOT` is set) automatic feedback logging + adaptation narrative — **no extra env vars required**.

## Input

User arguments:

`$ARGUMENTS`

## Execution plan

1. If no explicit input paths are provided, default to current workspace root.
2. Run pipeline script in `run` mode.
3. Enforce FTS-first retrieval with BM25 scoring.
4. Run vector retrieval adaptively (fallback default), merge results, and keep top **K** reranked chunks (plugin default **3** via `settings.json` → `defaultTopK`).
5. Apply **`relevanceFloor`** so low-scoring chunks are not summarized.
6. Run downstream shaping (routing tier, subagents when enabled, model-aware compression).
7. Compress before response handoff.

## Primary command (recommended: structured JSON)

Use **`--json`** so the model receives the complete packet: retrieval stats, token metrics, `claude_context` (plugin-shaped code + summary), and **`claude_context._context_adaptation`** (human-readable “why this context looks like this”) when running inside the plugin.

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/context_pipeline.py" run --inputs . --query "$ARGUMENTS" --hybrid-mode fallback --session-id default --top-k 3 --json
```

Plain-text packet only (smaller stdout, fewer fields):

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/context_pipeline.py" run --inputs . --query "$ARGUMENTS" --hybrid-mode fallback --session-id default --top-k 3
```

For stricter cost control, keep `--top-k 3` and a **specific** query (e.g. “Find the JWT validation function”) so irrelevant files score out before compression.

## Plug-and-play behavior

When **`CLAUDE_PLUGIN_ROOT`** is set (normal plugin install) and the index is an on-disk DB:

- **`TOKEN_REDUCER_FEEDBACK`** defaults to on → JSONL logging + workspace prune hints (`feedback.py`).
- **`TOKEN_REDUCER_ADAPTATION_BRIEF`** defaults on → `_context_adaptation` lines in `claude_context`.

Skipped automatically for ephemeral runs (`:memory:` DB or `session_id=auto`, e.g. stdin guard compression) so hooks do not spam logs.

**Opt out of extras:** set `TOKEN_REDUCER_MINIMAL=1`.

**Power users:** `TOKEN_REDUCER_ROUTING_DEBUG`, `TOKEN_REDUCER_PIPELINE_DEBUG`, `TOKEN_REDUCER_TIER_OVERRIDE`, etc. still work if set explicitly.

## Notes

- Prefer local pipeline to stay free-cost.
- ANN acceleration is used automatically when `hnswlib` + `numpy` are available.
- Use Context7 only when external library docs are needed.
- With `--json`, surface `estimated_token_savings`, `token_metrics`, and `claude_context` to the user.
- Large raw pasted query blobs are guarded; put logs/files in `--inputs`.
- Default chunk size, word budget, `top-k`, and relevance floor are loaded from the plugin **`settings.json`** when `CLAUDE_PLUGIN_ROOT` is set (repo `settings.json` when running from a checkout).
