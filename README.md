<div align="center">

# Token Reducer

### Local code search for Claude Code — hand the model the right 3 chunks instead of 30 whole files

[![Claude Code Plugin](https://img.shields.io/badge/Claude%20Code-Plugin-blueviolet?style=for-the-badge&logo=anthropic)](https://claude.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Release](https://img.shields.io/github/v/release/Madhan230205/token-reducer?style=for-the-badge)](https://github.com/Madhan230205/token-reducer/releases)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![SQLite](https://img.shields.io/badge/SQLite-FTS5-003B57?style=for-the-badge&logo=sqlite)](https://sqlite.org)

**Hybrid BM25 + embedding search over your repo, exposed as MCP tools Claude calls on demand. 100% local.**

[Install](#easy-install) • [MCP tools](#mcp-tools) • [Measure it](#measure-it-on-your-repo) • [Features](#features) • [Contributing](#contributing)

</div>

---

## The Problem

Claude Code explores a codebase with `Read` and `Grep`. Every whole-file read drops thousands of tokens into context, and most of them are misses while the model hunts for the right place. On a mid-sized repo that hunt is the bulk of what you pay for.

## The Solution

Token Reducer runs a small search engine next to Claude Code and exposes it as MCP tools:

- `search_code("where is the retry policy applied")` → the 3–6 most relevant chunks, with file path, symbol and score
- `read_file_slice(path, 120, 180)` → exact line ranges instead of entire files
- `get_context(task)` → a compressed, token-budgeted brief for a task
- `find_symbol(name)` → every place an identifier appears

Under the hood: tree-sitter AST chunking (functions/classes stay whole), SQLite FTS5 BM25 with stemming over code **and** file paths, ONNX MiniLM embeddings fused with reciprocal rank fusion, incremental re-indexing on every call. No network, no API keys, index stored under `~/.cache` — nothing is written into your repo.

```
┌────────────────┐   search_code()   ┌───────────────┐   3–6 chunks    ┌────────────┐
│  Claude Code   │ ────────────────▶ │ token-reducer │ ──────────────▶ │  Claude    │
│  (MCP client)  │ ◀──────────────── │  MCP server   │   ~400 tokens   │  answers   │
└────────────────┘                   └───────────────┘                 └────────────┘
                                      FTS5 BM25 + ONNX vectors, RRF, AST chunks
```

---

## Easy Install

### Option 1 — Claude Code `/plugin` Command (Recommended)

**Step 1:** Register the marketplace (one-time setup):
```
/plugin marketplace add Madhan230205/token-reducer
```

This registers the marketplace as `Madhan230205-token-reducer`.

**Step 2:** Install:
```
/plugin install claude-token-reducer@Madhan230205-token-reducer
```

For project-scoped install:
```
/plugin install claude-token-reducer@Madhan230205-token-reducer --scope project
```

> **Already ran Step 1 before?** Just run `/plugin install claude-token-reducer@Madhan230205-token-reducer` — no need to add the marketplace again.

---

### Option 2 — Git Clone (Manual)

```bash
# 1. Clone into your Claude plugins folder
git clone https://github.com/Madhan230205/token-reducer.git ~/.claude/plugins/token-reducer

# 2. Install dependencies (optional but recommended for best results)
pip install -r ~/.claude/plugins/token-reducer/requirements-optional.txt
```

> **Windows users**: Replace `~/.claude/plugins/` with `%USERPROFILE%\.claude\plugins\`

Then open `~/.claude/settings.json` and add:

```json
{
  "plugins": ["~/.claude/plugins/token-reducer"]
}
```

Restart Claude Code. Done.

---

**What `requirements-optional.txt` installs:**

| Package | Purpose |
|---------|---------|
| `sentence-transformers` | Neural embeddings for smarter retrieval |
| `hnswlib` / `faiss-cpu` | Fast approximate nearest-neighbor search |
| `tree-sitter` + language grammars | AST-based code chunking (Python, JS, TS, Go, Rust, Java, C/C++, Ruby) |

If you skip this step, Token Reducer still works using hash embeddings and regex chunking — no ML libraries required.

---

### Option 3 — Zero-Dependency Quick Start

No pip, no ML libs — runs immediately after cloning:

```bash
git clone https://github.com/Madhan230205/token-reducer.git
cd token-reducer
python scripts/context_pipeline.py run \
  --inputs ./src \
  --query "Find auth logic" \
  --embedding-backend hash \
  --db .cache/index.db
```

---

## MCP tools

The plugin registers an MCP server (`.mcp.json`) automatically. Claude Code will call these tools when they fit; the bundled skill nudges it to prefer `search_code` over blind `Read`/`Grep`.

| Tool | What it returns |
|------|-----------------|
| `search_code(query, top_k=6)` | Ranked chunks: `path:start-end  [function name]  score` + short preview |
| `get_context(task, top_k=3)` | JSON brief: summary, relevant files, relationships, trimmed code |
| `find_symbol(name)` | Files + snippets where an exact identifier occurs |
| `read_file_slice(path, start, end)` | Numbered lines for a range (end=0 → EOF) |
| `index_status()` / `reindex()` | Index stats / force an incremental sync |

First call on a repo builds the FTS index in seconds; dense vectors fill in on a background thread so search is never blocked.

**Use it from any MCP client** (Cursor, Windsurf, Zed, your own agent):

```json
{
  "mcpServers": {
    "token-reducer": {
      "command": "python",
      "args": ["/path/to/token-reducer/scripts/token_reducer_mcp.py"],
      "env": { "TOKEN_REDUCER_WORKSPACE": "/path/to/your/repo" }
    }
  }
}
```

Env knobs: `TOKEN_REDUCER_WORKSPACE` (default: cwd), `TOKEN_REDUCER_DB` (default: `~/.cache/token-reducer/workspaces/<hash>/index.db`), `TOKEN_REDUCER_EMBEDDING_BACKEND` (`onnx` | `hash`).

---

## Measure it on your repo

Compression ratios against a corpus pool (see [BENCHMARK.md](BENCHMARK.md)) are not the same as what a Claude Code session actually costs. `scripts/ab_eval.py` runs the same tasks through `claude -p` with and without the MCP server and reports real token usage, cost, turns and pass rate:

```bash
python scripts/ab_eval.py --repo /path/to/repo --tasks evals/ab_tasks.sample.json --repeats 2
```

Write 10–30 tasks for your own repo in the same JSON shape (`prompt` + `expect` substrings). If the B arm does not cut total tokens at equal pass rate, the tool is not earning its place in your setup — that is the number to trust.

**First measurement (honest):** on this repo itself (190 files, Sonnet, 6 "where is X" tasks, 1 repeat — [raw rows](evals/ab_results/2026-11-24-token-reducer-repo-sonnet.json)) the MCP arm did **not** win: cost −5%, total tokens +36% (mostly cached re-reads from one or two extra turns), pass rate 6/6 vs 5/5. Grep already answers identifier-style questions on a small repo cheaply, and the MCP tool schemas add to every turn's prefix. Where this tool should pay off — large repos, vague "how does X work" questions, non-identifier queries — is exactly what the harness is for. Run it on your repo before believing any savings number, including ours.

---

## Features

### Core Pipeline
- **Hybrid Retrieval** — BM25 + semantic vector search with intelligent fallback
- **AST-Based Chunking** — Tree-sitter parsing for Python, TypeScript, Go, Rust, Java, and more
- **TextRank Compression** — Graph-based sentence scoring for intelligent summarization
- **Sub-100ms Queries** — SQLite FTS5 + HNSW indexes for instant results
- **Local-First** — Everything runs on your machine, no external APIs

### LSP-Killer Features
- **Import Graph** — Automatically maps file dependencies without Language Server
- **2-Hop Symbol Expansion** — Auto "go-to-definition" for referenced functions
- **Diff Protocol** — SEARCH/REPLACE edit format with automatic application
- **Semantic Clustering** — Groups similar chunks to avoid redundancy

### Enterprise Ready
- **Fully Configurable** — 40+ tunable parameters in `settings.json`
- **Embedding Flexibility** — ML models or hash fallback (zero dependencies)
- **Query Caching** — Intelligent TTL-based caching for repeated queries
- **Session Memory** — Tracks context across conversation turns

---

## Documentation

### Showcase (decision traces & proof)

- [Why Token Reducer is different](docs/WHY_TOKEN_REDUCER.md) — one-page positioning  
- [Decision trace — JSON, UX, spine map](docs/showcase/decision-trace-star.md)  
- [Pipeline walkthrough](docs/showcase/pipeline-walkthrough.md) — Input → … → Output  
- [Benchmark proof snapshot](docs/showcase/benchmark-proof.md) — ties to [BENCHMARK.md](BENCHMARK.md)

### How It Works

```
Query → FTS(BM25) → (Vector fallback if needed) → Merge → Top 5 → Compress
```

Full pipeline:

```
PREPROCESS → INDEX → RETRIEVE → RE-RANK → COMPRESS → CONTEXT PACKET
```

### Basic Usage

```bash
# Index your codebase
python scripts/context_pipeline.py index --inputs ./src --db .cache/index.db

# Query with compression
python scripts/context_pipeline.py query \
  --query "How does authentication work?" \
  --db .cache/index.db \
  --json

# One-shot: index + query
python scripts/context_pipeline.py run \
  --inputs ./src \
  --query "Find the database connection logic" \
  --db .cache/index.db
```

### Configuration

All settings in `settings.json`:

```json
{
  "tokenReducer": {
    "chunkSizeWords": 220,
    "embeddingModel": "jinaai/jina-embeddings-v2-base-code",
    "hybridMode": "fallback",
    "astChunkingEnabled": true,
    "textRankEnabled": true,
    "lspFeatures": {
      "importGraphEnabled": true,
      "twoHopExpansionEnabled": true
    }
  }
}
```

<details>
<summary>Full Configuration Reference</summary>

| Setting | Default | Description |
|---------|---------|-------------|
| `chunkSizeWords` | 220 | Target words per chunk |
| `embeddingBackend` | "ml" | "ml" for neural, "hash" for zero-dep |
| `embeddingModel` | jina-v2-code | Code-optimized embeddings |
| `hybridMode` | "fallback" | "fallback" or "always" for vector |
| `astChunkingEnabled` | true | Use tree-sitter AST parsing |
| `textRankEnabled` | true | Graph-based sentence scoring |
| `importGraphEnabled` | true | Track file dependencies |
| `twoHopExpansionEnabled` | true | Auto-expand referenced symbols |
| `compressionWordBudget` | 350 | Max words in compressed output |

</details>

### Zero-Dependency Mode

Run without any ML libraries:

```bash
python scripts/context_pipeline.py run \
  --inputs ./src \
  --query "Find auth logic" \
  --embedding-backend hash \
  --db .cache/index.db
```

### Apply Code Edits

```bash
python scripts/apply_diff.py --input claude_response.txt --dir ./src
python scripts/apply_diff.py --input response.txt --dry-run
```

---

## Architecture

### Technology Stack

- **Storage**: SQLite with FTS5 + custom embeddings table
- **Chunking**: Tree-sitter AST parsing with regex fallback
- **Embeddings**: ONNX all-MiniLM-L6-v2 on CPU (batched; zero-dependency hash fallback)
- **ANN Search**: HNSW via hnswlib (with FAISS fallback)
- **Compression**: TextRank + query-relevance scoring

### Repository Structure

```text
token-reducer/
├── .claude-plugin/plugin.json
├── .mcp.json                 # registers the MCP server
├── .env.example
├── settings.json
├── requirements-optional.txt
├── scripts/
│   ├── token_reducer_mcp.py  # MCP server entry
│   ├── ab_eval.py            # A/B token-usage eval via `claude -p`
│   └── token_reducer/        # pipeline package
├── hooks/
├── commands/
├── agents/
├── skills/
└── evals/
```

---

## Contributing

If anyone is interested in contributing, this project is open to contributions.
Please see [contribute.md](contribute.md) for contribution guidelines.

```bash
git clone https://github.com/Madhan230205/token-reducer.git
cd token-reducer
pip install -e ".[dev]"
python scripts/context_pipeline.py self-test
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) for AST parsing
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [SQLite FTS5](https://sqlite.org/fts5.html) for blazing-fast text search
- [hnswlib](https://github.com/nmslib/hnswlib) for approximate nearest neighbors

---

<div align="center">

**Star this repo if Token Reducer saves you money!**

[Report Bug](https://github.com/Madhan230205/token-reducer/issues) • [Request Feature](https://github.com/Madhan230205/token-reducer/issues) • [Discussions](https://github.com/Madhan230205/token-reducer/discussions)

</div>
