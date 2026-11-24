---
name: token-reducer
description: Use this skill whenever you need to find where something lives in the codebase, or the user needs to reduce context bloat, lower token usage, summarize large code/docs corpora, run FTS plus embeddings retrieval, rerank top chunks, and produce compact context packets before implementation work. Also trigger when users say context is too long, reduce tokens, optimize prompt size, retrieve top chunks, compress context, or ask for cost-saving prompt workflows.
license: MIT
compatibility: Python 3.11+ with SQLite FTS5. Fully local and free by default. Ships an MCP server (search_code, get_context, find_symbol, read_file_slice).
allowed-tools: [Read, Glob, Grep, Bash, Task, mcp__plugin_claude-token-reducer_token-reducer, mcp__token-reducer]
metadata:
  author: Madhan230205
  version: "1.0.0"
---

# token-reducer

Cut context size without cutting answer quality.

## Adaptive context (invisible to the user)

The pipeline maps each **query + intent + corpus tier** to a **context strategy** (merge depth, prune width, whether to run fusion/neighborhood, and a single **attention frame** for the model). That is **not** a fixed “always N chunks” shape: short or conversational prompts get a lighter path; bug-fix and navigation prompts bias retrieval and framing without exposing internal stage names. The goal is better answers with less noise — the user should notice sharper reasoning, not a separate “reducer” gadget.

## Why token use still spikes

Claude Code often answers code questions with native **Read** / **Grep** on whole files, which loads raw text into the model and **bypasses** this pipeline. Long **chat history** is re-sent every turn, so costs compound even when code is compressed.

## Workflow (prefer this order)

1. **Locate with MCP first.** Before `Read`-ing whole files or broad `Grep`s, call the token-reducer MCP tool `search_code` (full name `mcp__plugin_claude-token-reducer_token-reducer__search_code` when installed as a plugin) with a specific query (what the code *does*, plus any identifiers you know). It returns the few most relevant chunks with file + symbol. Use `find_symbol` for an exact identifier.
2. **Read narrowly.** Once you know the file, use `read_file_slice(path, start, end)` for the lines you need instead of the whole file. Fall back to native `Read` only when you truly need the full file (e.g. you are about to edit it).
3. **Brief for bigger tasks.** `get_context(task)` returns a compressed brief (summary, relevant files, relationships, trimmed code) — good as the first step of a refactor or bug hunt.
4. **Do not paste large code or logs into chat** — that bypasses retrieval and burns tokens. Point at paths instead.
5. **CLI fallback** (same engine, when MCP is unavailable):

   `python "${CLAUDE_PLUGIN_ROOT}/scripts/context_pipeline.py" run --inputs ./src --query "Locate JWT validation" --top-k 3 --json`

6. **Session hygiene**: around **10** turns the hook suggests `/compact`; by **40–50** turns start a new chat for coding after planning.

## Pipeline (what the tool does)

1. Preprocess large/noisy corpus into overlap-aware chunks (size/overlap from `settings.json` → `chunkSizeWords` / `chunkOverlapWords`).
2. Index chunks into SQLite FTS5 and local embeddings.
3. Retrieve with BM25-first policy; vector fallback when configured.
4. Merge + rerank; keep top **K** (default **3** from `defaultTopK` in `settings.json`).
5. Compress with TextRank/word budget; drop chunks below **`relevanceFloor`**.
6. Emit citation-rich packet + savings telemetry.

## Commands

- End-to-end run (defaults from plugin `settings.json`; override flags as needed):

  `python "${CLAUDE_PLUGIN_ROOT}/scripts/context_pipeline.py" run --inputs . --query "${ARGUMENTS}" --hybrid-mode fallback --top-k 3 --json`

- Self-test:

  `python "${CLAUDE_PLUGIN_ROOT}/scripts/context_pipeline.py" self-test`

## Tuning (plugin `settings.json` under `tokenReducer`)

- **`compressionWordBudget`** — lower for shorter summaries (e.g. 150).
- **`chunkSizeWords`** / **`chunkOverlapWords`** — smaller chunks before compression (e.g. 100 / 20).
- **`defaultTopK`** — fewer final chunks (e.g. 3).
- **`relevanceFloor`** — higher values drop more weak chunks before summarization (e.g. 0.18).

Session reminders: top-level **`promptGuard`** (`autoCompactTurn`, `autoResetTurn`, `criticalResetTurn`, `reminderTurns`) plus `historyCompactReminderTurns` inside `tokenReducer`.

## Decision trace (reviewer / replay path)

The pipeline emits a **single canonical JSON** — **`decision_trace_v2`** — from the same spine + run state as hybrid explainability. It appears as **`claude_context.decision_trace`** and **`ContextPacket.decision_trace`** (and on disk under `--trace-dir` / `TOKEN_REDUCER_TRACE_DIR` as `decision_trace.json`). **Flow:** query → decision trace → chunk refs (`selected_chunks`) → compressed bullets / model answer. **`summary.txt`** next to the trace is a deterministic template (no LLM). For the previous `focus_line` / `agent_trace` / `chunk_transparency` packet fields, set **`TOKEN_REDUCER_LEGACY_PACKET_EXPLAIN=1`**.

## Deep References

- `./references/implementation-guide.md`
- `./references/context7-integration.md`
