---
name: token-reducer
description: Use this skill whenever the user needs to reduce context bloat, lower token usage, summarize large code/docs corpora, run FTS plus embeddings retrieval, rerank top chunks, and produce compact context packets before implementation work. Also trigger when users say context is too long, reduce tokens, optimize prompt size, retrieve top chunks, compress context, or ask for cost-saving prompt workflows.
license: MIT
compatibility: Python 3.10+ with SQLite FTS5. Fully local and free by default. Context7 MCP is optional.
allowed-tools: [Read, Glob, Grep, Bash, Task]
metadata:
  author: Madv6
  version: "1.0.0"
---

# token-reducer

Cut context size without cutting answer quality.

## Workflow

1. Preprocess large/noisy corpus into overlap-aware chunks.
2. Index chunks into SQLite FTS5 and local embeddings.
3. Retrieve with BM25-first policy.
4. Run vector retrieval in fallback mode by default.
5. Merge + rerank candidates; keep top 3-5.
6. Compress into citation-rich packet and output savings telemetry.

## Commands

- End-to-end run:
  - `python "${CLAUDE_PLUGIN_ROOT}/scripts/context_pipeline.py" run --inputs . --query "${ARGUMENTS}" --hybrid-mode fallback --top-k 5`
- Self-test:
  - `python "${CLAUDE_PLUGIN_ROOT}/scripts/context_pipeline.py" self-test`

## Deep References

- `./references/implementation-guide.md`
- `./references/context7-integration.md`
