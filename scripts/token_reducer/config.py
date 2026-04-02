from __future__ import annotations

import re
from pathlib import Path

DEFAULT_DB_PATH = Path(".cache/token-reducer/index.db")
DEFAULT_CHUNK_SIZE = 220
DEFAULT_CHUNK_OVERLAP = 40
DEFAULT_DIMENSIONS = 256
DEFAULT_FTS_K = 12
DEFAULT_VECTOR_K = 20
DEFAULT_TOP_K = 5
DEFAULT_MIN_FTS_HITS = 3
DEFAULT_WORD_BUDGET = 350
DEFAULT_HYBRID_MODE = "fallback"
DEFAULT_RETRIEVAL_MODE = "compact"
DEFAULT_EMBEDDING_BACKEND = "hash"
DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"
DEFAULT_ANN_ENGINE = "hnsw"
DEFAULT_ANN_EF_SEARCH = 160
DEFAULT_QUERY_CACHE_TTL_SECONDS = 900

# Adaptive retrieval tiers — determined at runtime from indexed chunk count.
# Small  (<  TIER_SMALL_CHUNKS)  → FTS5 only; no embeddings used for retrieval, no ANN built.
# Medium (< TIER_MEDIUM_CHUNKS)  → FTS5 primary + hash-embedding fallback; no ANN built.
# Large  (>= TIER_MEDIUM_CHUNKS) → Full hybrid: FTS5 + embeddings + ANN (HNSW).
ADAPTIVE_TIER_SMALL_CHUNKS = 200
ADAPTIVE_TIER_MEDIUM_CHUNKS = 2000
MIN_CHUNK_WORDS = 150
MAX_CHUNK_WORDS = 300
MAX_CHUNK_TOKEN_ESTIMATE = 400
MAX_QUERY_WORDS = 500
MAX_QUERY_LINES = 80
MAX_COMPRESSED_TO_SELECTED_RATIO = 0.60

# Scoring weights - configurable via settings.json -> scoringWeights
# These defaults can be overridden at runtime
DEFAULT_SCORING_WEIGHTS = {
    "fts_lexical_rank_weight": 0.35,
    "fts_bm25_weight": 0.65,
    "final_fts_weight": 0.50,
    "final_vector_weight": 0.35,
    "final_overlap_weight": 0.15,
    "sentence_length_bonus_weight": 0.10,
    "sentence_length_normalizer": 24.0,
    "char_ngram_weight": 0.35,
    "textrank_weight": 0.50,
    "query_relevance_weight": 0.35,
}

# When hash embeddings are used, skip vector retrieval entirely since
# hash embeddings provide lexical similarity (redundant with FTS5/BM25)
DEFAULT_HASH_EMBEDDING_SKIP_VECTOR = True

_EMBEDDING_MODEL_CACHE: dict[str, object] = {}
_EMBEDDING_VECTOR_CACHE: dict[str, list[float]] = {}
_SCORING_WEIGHTS: dict[str, float] = DEFAULT_SCORING_WEIGHTS.copy()
_HASH_EMBEDDING_SKIP_VECTOR: bool = DEFAULT_HASH_EMBEDDING_SKIP_VECTOR


def configure_scoring_weights(weights: dict[str, float] | None = None) -> None:
    """Update scoring weights from external configuration."""
    global _SCORING_WEIGHTS
    if weights:
        for key, value in weights.items():
            # Convert camelCase keys from JSON to snake_case
            snake_key = _camel_to_snake(key)
            if snake_key in DEFAULT_SCORING_WEIGHTS:
                _SCORING_WEIGHTS[snake_key] = float(value)


def configure_hash_skip_vector(skip: bool) -> None:
    """Configure whether to skip vector retrieval when using hash embeddings."""
    global _HASH_EMBEDDING_SKIP_VECTOR
    _HASH_EMBEDDING_SKIP_VECTOR = skip


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    import re
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def get_weight(key: str) -> float:
    """Get a scoring weight by key."""
    return _SCORING_WEIGHTS.get(key, DEFAULT_SCORING_WEIGHTS.get(key, 0.0))


def should_skip_vector_for_hash() -> bool:
    """Check if vector retrieval should be skipped when using hash embeddings."""
    return _HASH_EMBEDDING_SKIP_VECTOR


TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".rst",
    ".adoc",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".sh",
    ".ps1",
    ".java",
    ".kt",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".cs",
    ".php",
    ".rb",
    ".swift",
    ".sql",
    ".xml",
    ".html",
    ".css",
    ".scss",
    ".env",
}

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go", ".rs",
    ".c", ".h", ".cpp", ".cs", ".php", ".rb", ".swift", ".sh", ".ps1",
}

IGNORED_FILENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "composer.lock",
    "Gemfile.lock",
    "Cargo.lock",
    "poetry.lock",
    "Pipfile.lock",
    "bun.lockb",
    "shrinkwrap.json",
    "npm-shrinkwrap.json",
}

# Import graph extraction patterns
_IMPORT_PATTERNS: dict[str, re.Pattern[str]] = {
    ".py": re.compile(
        r"^(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE
    ),
    ".js": re.compile(
        r"(?:import\s+.*?from\s+['\"]([^'\"]+)['\"]|require\s*\(\s*['\"]([^'\"]+)['\"]\s*\))", re.MULTILINE
    ),
    ".ts": re.compile(
        r"(?:import\s+.*?from\s+['\"]([^'\"]+)['\"]|require\s*\(\s*['\"]([^'\"]+)['\"]\s*\))", re.MULTILINE
    ),
    ".tsx": re.compile(
        r"(?:import\s+.*?from\s+['\"]([^'\"]+)['\"]|require\s*\(\s*['\"]([^'\"]+)['\"]\s*\))", re.MULTILINE
    ),
    ".jsx": re.compile(
        r"(?:import\s+.*?from\s+['\"]([^'\"]+)['\"]|require\s*\(\s*['\"]([^'\"]+)['\"]\s*\))", re.MULTILINE
    ),
    ".go": re.compile(r'import\s+(?:\(\s*)?["\']([^"\']+)["\']', re.MULTILINE),
    ".rs": re.compile(r"(?:use\s+([\w:]+)|mod\s+(\w+))", re.MULTILINE),
    ".java": re.compile(r"import\s+([\w.]+);", re.MULTILINE),
    ".c": re.compile(r'#include\s*[<"]([^>"]+)[>"]', re.MULTILINE),
    ".h": re.compile(r'#include\s*[<"]([^>"]+)[>"]', re.MULTILINE),
    ".cpp": re.compile(r'#include\s*[<"]([^>"]+)[>"]', re.MULTILINE),
    ".rb": re.compile(r"(?:require\s+['\"]([^'\"]+)['\"]|require_relative\s+['\"]([^'\"]+)['\"])", re.MULTILINE),
}

# Function call extraction pattern (language-agnostic)
_FUNCTION_CALL_PATTERN = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", re.MULTILINE)
