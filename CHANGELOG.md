# Changelog

## 1.0.0 - 2026-04-01

- Initial public release of `token-reducer`.
- Free, local-first token reduction pipeline:
  - preprocessing and chunking
  - SQLite FTS5 retrieval with BM25 ranking
  - adaptive hybrid retrieval (fallback default)
  - ANN acceleration (HNSW) when optional deps are available
  - reranking and top-3 to top-5 selection
  - compression with token-efficiency guardrails
  - query/result caching and lightweight session memory
- Added `UserPromptSubmit` hook guard to reduce prompt bloat risk.
- Added marketplace-ready plugin manifest and install/publish docs.
