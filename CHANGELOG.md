# Changelog

## 1.5.0 - 2026-05-02

### Added — Adaptive context, agents, transparency, learning hooks

- **Query → context strategy** (`context_strategy.py`): maps intent, query shape, retrieval tier, and vector use to `merge_cap`, `prune_k`, optional skips for fusion/neighborhood, and an internal **attention frame** (editorial line folded into the plugin summary, not exposed as pipeline jargon).
- **Orchestrator integration**: strategy is chosen after retrieval; merge stage and subagents respect caps/skips; Claude plugin payload keeps framing **inside `summary`** (no `attention_frame` / `context_shape` keys).
- **Subagent layer** (`subagents/`): deterministic **coordinator** (`run_memory`: decomposition steps, `focus_paths`, `focus_terms`), **merge_streams** (path stream + term stream + remainder interleaved), **filter**, **ranking** (path/term boosts + feedback boosts), **variance** (deterministic mid-list nudge from underrepresented files), **fusion**, **budget**.
- **Coordination** (`coordination.py`): path/term extraction, task decomposition strings, `merge_subtask_streams`, `diversify_chunk_order`.
- **Feedback & adaptation** (`feedback.py`): JSONL logging; `read_strategy_prune_adjustments` from recent logs; optional **workspace EMA** at `<workspace>/.token-reducer/prune_ema.json` (updated when `TOKEN_REDUCER_FEEDBACK` is set); orchestrator blends log + EMA into `prune_k` nudge.
- **Pipeline logging**: `extra.strategy_id` and **`chunk_trace`** (`chunk_id`, `final_score`) for future ranking / analytics.
- **User-facing packet copy** (`context_explain.py`): **`focus_line`** (“kept N excerpts from these files for your {intent}…”), **`agent_trace`** (intent → retrieval → strategy → context_agents → compress → validate), **`chunk_transparency`** (per-chunk signal mix + scores). Wired in `build_packet_from_state`; text packet includes `focus:` after `query:`.
- **`ContextPacket` fields**: `focus_line`, `agent_trace`, `chunk_transparency` (structured trust layer alongside existing `candidates`).

### Changed

- **Hooks** (`userprompt_guard.py`): softer, outcome-first system messages (less “token” / plugin jargon on success paths); hard block still actionable.
- **`.gitignore`**: SQLite / HNSW / FAISS-style artifacts and `.token-reducer/` workspace cache dir for cleaner clones.
- **Skill** (`skills/token-reducer/SKILL.md`): adaptive shaping and invisible framing notes.

### Fixed

- **`ContextPacket` / `Symbol` model**: restored correct class boundary (avoid Pydantic treating packet fields as `Symbol`).

---

## 1.4.0 - 2026-04-01

### Added - LSP-Killer Architecture
- **Import Graph ("Fake LSP")** - Extracts and stores file dependencies during indexing
  - New `file_dependencies` SQLite table maps imports between files
  - `extract_imports()` parses Python, JS/TS, Go, Rust, Java, C/C++, Ruby imports
  - `resolve_import_to_file()` maps import paths to actual indexed files
  - `fetch_imported_context()` auto-fetches relevant chunks from imported files
- **2-Hop Symbol Expansion** - Auto "go-to-definition" for function calls
  - New `symbol_index` SQLite table stores function/class definitions
  - `extract_function_calls()` finds function calls in selected chunks
  - `lookup_symbol_definition()` fetches definitions from index
  - `expand_symbols_two_hop()` appends referenced definitions to context
  - Eliminates need for full Language Server
- **Diff Protocol** - Edit instructions in context packet
  - SEARCH/REPLACE format for Claude to output code changes
  - `scripts/apply_diff.py` - Standalone script to apply diff blocks
  - Supports dry-run mode and JSON output
- **Symbol extraction** for Python, JS/TS, Go, Rust with signature capture

### Changed
- `build_packet()` now includes `referenced_symbols` and `imported_context` sections
- Context packets include edit protocol instructions

### Architecture
This release transforms token-reducer from a passive context retriever into an active code understanding tool that:
1. Knows which files import each other (Import Graph)
2. Auto-expands function definitions without Language Server (2-Hop)
3. Enables direct code editing via diff protocol

## 1.3.0 - 2026-04-01

### Added
- **TextRank intelligent compression** - Graph-based sentence scoring for semantic summarization
  - Sentences scored by centrality in semantic similarity graph
  - Combined with query relevance for optimal extraction
  - Configurable via `textRankEnabled`, `textRankDamping`, `textRankIterations`
- **Benchmark command** (`context_pipeline.py benchmark`) for proof-of-value metrics
  - Before/after token comparison
  - Compression ratio measurements
  - Latency benchmarks (indexing + query)
  - Cost savings analysis ($/query estimates)
  - JSON + human-readable output
- **Semantic chunk clustering** - k-means clustering on embeddings for diverse selection
  - Groups similar chunks to avoid redundancy
  - Configurable via `semanticClusteringEnabled`, `semanticClusterCount`
- **ANN tuning parameters** exposed in settings:
  - `annEfConstruction` - Build-time accuracy/speed tradeoff
  - `annM` - Graph connectivity (memory/accuracy tradeoff)
- **Embedding cache configuration** - LRU cache with TTL for embeddings
  - `embeddingCache.maxSize`, `embeddingCache.ttlSeconds`, `embeddingCache.strategy`
- New scoring weights: `textRankWeight`, `queryRelevanceWeight`

### Changed
- Prose compression now uses TextRank + query relevance (was: query overlap only)
- Sentence scoring formula: `0.5 * textrank + 0.35 * query_signal + 0.15 * length_bonus`

## 1.2.0 - 2026-04-01

### Added
- **Configurable scoring weights** via `settings.json -> scoringWeights`
  - `ftsLexicalRankWeight` / `ftsBm25Weight` - FTS score composition
  - `finalFtsWeight` / `finalVectorWeight` / `finalOverlapWeight` - final ranking blend
  - `sentenceLengthBonusWeight` / `sentenceLengthNormalizer` - prose compression tuning
  - `charNgramWeight` - hash embedding n-gram contribution
- Added `hashEmbeddingSkipVector` setting (default: true) to skip redundant vector retrieval when using hash embeddings

### Changed
- **Hash embedding behavior**: When `hashEmbeddingSkipVector` is true, vector retrieval is skipped entirely for hash backend since it provides lexical similarity redundant with FTS5/BM25
- All hardcoded scoring weights now read from `_SCORING_WEIGHTS` dictionary, configurable at runtime

### Fixed
- Removed noise from hash embedding vector retrieval by skipping it by default (FTS5 already captures lexical overlap more accurately)

## 1.1.0 - 2026-04-01

### Added
- **True AST-based code chunking** using tree-sitter for semantic code boundaries
  - Parses Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, Ruby via tree-sitter
  - Extracts function definitions, class declarations, interfaces, structs as semantic units
  - Falls back to regex patterns when tree-sitter unavailable
- Added tree-sitter grammars to `requirements-optional.txt`

### Changed
- **Default embedding model** changed from `sentence-transformers/all-MiniLM-L6-v2` to `jinaai/jina-embeddings-v2-base-code`
  - Code-native embeddings produce better semantic similarity for source code
  - Significantly improves vector retrieval accuracy for code queries
- Updated `settings.json` with new code-native embedding model default

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
