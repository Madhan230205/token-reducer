---
name: token-reducer
description: Use this skill whenever the user needs to reduce context bloat, lower token usage, summarize large code/docs corpora, run FTS plus embeddings retrieval, rerank top chunks, and produce compact context packets before implementation work. Also trigger when users say context is too long, reduce tokens, optimize prompt size, retrieve top chunks, compress context, or ask for cost-saving prompt workflows.
license: MIT
compatibility: Python 3.10+ with SQLite FTS5. Fully local and free by default. Context7 MCP is optional.
allowed-tools: [Read, Glob, Grep, Bash, Task]
metadata:
  author: Madhan230205
  version: "1.0.0"
---

# token-reducer

Cut context size without cutting answer quality.

## Why token use still spikes

Claude Code often answers code questions with native **Read** / **Grep** on whole files, which loads raw text into the model and **bypasses** this pipeline. Long **chat history** is re-sent every turn, so costs compound even when code is compressed.

## Workflow (prefer this order)

1. **Do not paste large code or logs into chat** — that bypasses reduction and burns tokens.
2. **Run the slash command first** so the pipeline runs before reasoning, for example: use `/token-reducer` with a short objective and paths (defaults come from plugin `settings.json`: small chunks, low `--top-k`, word budget, `relevanceFloor`).
3. **CLI (same pipeline)** when you want a packet on disk or in a script:

   `python "${CLAUDE_PLUGIN_ROOT}/scripts/context_pipeline.py" run --inputs ./src --query "Locate JWT validation" --top-k 3`

   Use a **specific query** (not “auth stuff”) so low-scoring chunks are dropped by the **relevance floor** before summarization.

4. **Session hygiene**: around **10** turns the hook suggests `/compact`; by **40–50** turns start a **new chat** for coding after planning.

## Pipeline (what the tool does)

1. Preprocess large/noisy corpus into overlap-aware chunks (size/overlap from `settings.json` → `chunkSizeWords` / `chunkOverlapWords`).
2. Index chunks into SQLite FTS5 and local embeddings.
3. Retrieve with BM25-first policy; vector fallback when configured.
4. Merge + rerank; keep top **K** (default **3** from `defaultTopK` in `settings.json`).
5. Compress with TextRank/word budget; drop chunks below **`relevanceFloor`**.
6. Emit citation-rich packet + savings telemetry.

## Commands

- End-to-end run (defaults from plugin `settings.json`; override flags as needed):

  `python "${CLAUDE_PLUGIN_ROOT}/scripts/context_pipeline.py" run --inputs . --query "${ARGUMENTS}" --hybrid-mode fallback --top-k 3`

- Self-test:

  `python "${CLAUDE_PLUGIN_ROOT}/scripts/context_pipeline.py" self-test`

## Tuning (plugin `settings.json` under `tokenReducer`)

- **`compressionWordBudget`** — lower for shorter summaries (e.g. 150).
- **`chunkSizeWords`** / **`chunkOverlapWords`** — smaller chunks before compression (e.g. 100 / 20).
- **`defaultTopK`** — fewer final chunks (e.g. 3).
- **`relevanceFloor`** — higher values drop more weak chunks before summarization (e.g. 0.18).

Session reminders: top-level **`promptGuard`** (`autoCompactTurn`, `autoResetTurn`, `criticalResetTurn`, `reminderTurns`) plus `historyCompactReminderTurns` inside `tokenReducer`.

## Deep References

- `./references/implementation-guide.md`
- `./references/context7-integration.md`
