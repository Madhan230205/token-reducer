<div align="center">

# Token Reducer

### Cut Claude API costs by 90%+ with intelligent context compression

[![Claude Code Plugin](https://img.shields.io/badge/Claude%20Code-Plugin-blueviolet?style=for-the-badge&logo=anthropic)](https://claude.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![SQLite](https://img.shields.io/badge/SQLite-FTS5-003B57?style=for-the-badge&logo=sqlite)](https://sqlite.org)

**The open-source alternative to expensive context management tools.**

[Easy Install](#-easy-install) • [Features](#-features) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## The Problem

Every time you use Claude with a large codebase, you're paying for **thousands of tokens** that aren't relevant to your query. Most context management tools either:

- Send everything (expensive)
- Truncate blindly (loses important context)
- Require heavy Language Servers (slow, resource-intensive)

## The Solution

**Token Reducer** is a local-first, intelligent context compression pipeline that:

- Reduces tokens by 90-98% while preserving semantic relevance
- Runs entirely locally — no API calls, no data leaving your machine
- Works in milliseconds — faster than Language Server alternatives
- Understands code semantically — AST parsing, not just text matching

```
┌─────────────────┐     ┌───────────────┐     ┌──────────────────┐
│  Your Codebase  │────▶│ Token Reducer │────▶│  Compressed      │
│  (50,000 tokens)│     │   Pipeline    │     │  Context (500t)  │
└─────────────────┘     └───────────────┘     └──────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  - AST Chunking   │
                    │  - BM25 + Vector  │
                    │  - TextRank       │
                    │  - Import Graph   │
                    │  - 2-Hop Symbols  │
                    └───────────────────┘
```

---

## Easy Install

### Option 1 — Git Clone (Recommended for Everyone)

```bash
# 1. Clone into your Claude plugins folder
git clone https://github.com/Madhan230205/token-reducer.git ~/.claude/plugins/token-reducer

# 2. Install dependencies (optional but recommended for best results)
pip install -r ~/.claude/plugins/token-reducer/requirements-optional.txt
```

> **Windows users**: Replace `~/.claude/plugins/` with `%USERPROFILE%\.claude\plugins\`

Then tell Claude Code where the plugin lives — open your `~/.claude/settings.json` and add:

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

### Option 2 — Claude Code Marketplace

If you have a Claude Code marketplace configured:

```
/plugin install github:Madhan230205/token-reducer
```

For project-scoped install:

```
/plugin install github:Madhan230205/token-reducer --scope project
```

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
- **Embeddings**: Jina Code v2 (or zero-dependency hash embeddings)
- **ANN Search**: HNSW via hnswlib (with FAISS fallback)
- **Compression**: TextRank + query-relevance scoring

### Repository Structure

```text
token-reducer/
├── .claude-plugin/plugin.json
├── .mcp.json
├── .env.example
├── settings.json
├── requirements-optional.txt
├── scripts/
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
