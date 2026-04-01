#!/usr/bin/env python3
"""Local-first context slimming pipeline for Claude Code plugins.

Pipeline stages:
1) Preprocess (noise removal + chunking)
2) Index (SQLite FTS5 + local embeddings)
3) Retrieve (FTS first)
4) Vector retrieval (fallback by default; always mode optional)
5) Merge + Re-rank (top 3-5, default top 5)
6) Compress and emit a Claude-ready context packet
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import blake2b
from pathlib import Path
from typing import Iterable, Sequence


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
DEFAULT_EMBEDDING_BACKEND = "ml"
DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"
DEFAULT_ANN_ENGINE = "hnsw"
DEFAULT_ANN_EF_SEARCH = 160
DEFAULT_QUERY_CACHE_TTL_SECONDS = 900
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


def extract_imports(text: str, source: str) -> list[str]:
    """Extract import/require statements from source code."""
    ext = Path(source).suffix.lower()
    pattern = _IMPORT_PATTERNS.get(ext)
    if pattern is None:
        return []
    
    imports: list[str] = []
    for match in pattern.finditer(text):
        for group in match.groups():
            if group:
                imports.append(group.strip())
    return imports


def extract_function_calls(text: str) -> list[str]:
    """Extract function/method call names from code text."""
    # Filter out common keywords and built-ins
    keywords = {
        "if", "else", "for", "while", "return", "function", "def", "class",
        "import", "from", "try", "except", "catch", "finally", "with", "as",
        "print", "len", "str", "int", "float", "list", "dict", "set", "tuple",
        "True", "False", "None", "self", "this", "new", "const", "let", "var",
    }
    calls = []
    for match in _FUNCTION_CALL_PATTERN.finditer(text):
        name = match.group(1)
        if name not in keywords and not name.startswith("_"):
            calls.append(name)
    return list(set(calls))  # Deduplicate


def resolve_import_to_file(import_path: str, source_file: str, indexed_files: set[str]) -> str | None:
    """Attempt to resolve an import path to an actual indexed file."""
    source_dir = Path(source_file).parent
    
    # Common resolution strategies
    candidates = [
        # Relative import
        str(source_dir / f"{import_path}.py"),
        str(source_dir / f"{import_path}.ts"),
        str(source_dir / f"{import_path}.js"),
        str(source_dir / import_path / "__init__.py"),
        str(source_dir / import_path / "index.ts"),
        str(source_dir / import_path / "index.js"),
        # Convert dot notation to path
        str(source_dir / import_path.replace(".", "/") + ".py"),
        str(source_dir / import_path.replace(".", "/") + ".ts"),
    ]
    
    for candidate in candidates:
        normalized = str(Path(candidate).resolve())
        if normalized in indexed_files:
            return normalized
    
    return None


def index_file_dependencies(
    conn: sqlite3.Connection,
    file_path: str,
    content: str,
    indexed_files: set[str],
) -> int:
    """Extract and store import dependencies for a file."""
    imports = extract_imports(content, file_path)
    count = 0
    
    for import_path in imports:
        resolved = resolve_import_to_file(import_path, file_path, indexed_files)
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO file_dependencies (source_file, target_import, resolved_file)
                VALUES (?, ?, ?)
                """,
                (file_path, import_path, resolved),
            )
            count += 1
        except Exception:
            continue
    
    return count


def extract_symbols_from_chunk(text: str, source: str) -> list[dict]:
    """Extract function/class symbols from a code chunk using AST or regex."""
    symbols: list[dict] = []
    ext = Path(source).suffix.lower()
    
    # Use regex patterns as fallback
    if ext == ".py":
        # Python: def func_name(...) or class ClassName
        for match in re.finditer(r"^(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "function",
                "signature": f"def {match.group(1)}({match.group(2)})",
            })
        for match in re.finditer(r"^class\s+(\w+)(?:\(([^)]*)\))?:", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "class",
                "signature": f"class {match.group(1)}",
            })
    elif ext in {".js", ".ts", ".tsx", ".jsx"}:
        # JS/TS: function name() or const name = () =>
        for match in re.finditer(r"(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "function",
                "signature": f"function {match.group(1)}({match.group(2)})",
            })
        for match in re.finditer(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "function",
                "signature": f"const {match.group(1)} = () =>",
            })
        for match in re.finditer(r"class\s+(\w+)(?:\s+extends\s+\w+)?", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "class",
                "signature": f"class {match.group(1)}",
            })
    elif ext == ".go":
        for match in re.finditer(r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(([^)]*)\)", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "function",
                "signature": f"func {match.group(1)}({match.group(2)})",
            })
        for match in re.finditer(r"type\s+(\w+)\s+struct", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "struct",
                "signature": f"type {match.group(1)} struct",
            })
    elif ext == ".rs":
        for match in re.finditer(r"(?:pub\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "function",
                "signature": f"fn {match.group(1)}({match.group(2)})",
            })
        for match in re.finditer(r"(?:pub\s+)?struct\s+(\w+)", text, re.MULTILINE):
            symbols.append({
                "name": match.group(1),
                "type": "struct",
                "signature": f"struct {match.group(1)}",
            })
    
    return symbols


def index_symbols(
    conn: sqlite3.Connection,
    file_path: str,
    chunk_id: int,
    chunk_text: str,
) -> int:
    """Extract and store symbols from a chunk."""
    symbols = extract_symbols_from_chunk(chunk_text, file_path)
    count = 0
    
    for symbol in symbols:
        try:
            conn.execute(
                """
                INSERT INTO symbol_index (file_path, symbol_name, symbol_type, chunk_id, signature)
                VALUES (?, ?, ?, ?, ?)
                """,
                (file_path, symbol["name"], symbol["type"], chunk_id, symbol.get("signature")),
            )
            count += 1
        except Exception:
            continue
    
    return count


def get_imported_files(conn: sqlite3.Connection, source_file: str) -> list[str]:
    """Get list of files imported by a source file."""
    rows = conn.execute(
        """
        SELECT DISTINCT resolved_file FROM file_dependencies
        WHERE source_file = ? AND resolved_file IS NOT NULL
        """,
        (source_file,),
    ).fetchall()
    return [str(row[0]) for row in rows]


def lookup_symbol_definition(conn: sqlite3.Connection, symbol_name: str, limit: int = 3) -> list[dict]:
    """Look up symbol definitions by name (for 2-hop expansion)."""
    rows = conn.execute(
        """
        SELECT s.file_path, s.symbol_type, s.signature, c.text
        FROM symbol_index s
        JOIN chunks c ON s.chunk_id = c.id
        WHERE s.symbol_name = ?
        LIMIT ?
        """,
        (symbol_name, limit),
    ).fetchall()
    
    return [
        {
            "file": row[0],
            "type": row[1],
            "signature": row[2],
            "definition": row[3][:500] if row[3] else None,  # Truncate long definitions
        }
        for row in rows
    ]


def expand_symbols_two_hop(
    conn: sqlite3.Connection,
    candidates: list[Candidate],
    max_expansions: int = 5,
) -> list[dict]:
    """Perform 2-hop symbol expansion on selected candidates.
    
    Extracts function calls from selected chunks and fetches their definitions.
    This simulates LSP "go to definition" functionality.
    """
    referenced_symbols: list[dict] = []
    seen_symbols: set[str] = set()
    
    for candidate in candidates[:3]:  # Only expand top 3 chunks
        calls = extract_function_calls(candidate.text)
        
        for call_name in calls:
            if call_name in seen_symbols:
                continue
            if len(referenced_symbols) >= max_expansions:
                break
            
            definitions = lookup_symbol_definition(conn, call_name, limit=1)
            if definitions:
                seen_symbols.add(call_name)
                referenced_symbols.append({
                    "symbol": call_name,
                    "from_chunk": candidate.source,
                    **definitions[0],
                })
    
    return referenced_symbols


def fetch_imported_context(
    conn: sqlite3.Connection,
    selected_sources: list[str],
    query: str,
    limit_per_file: int = 2,
) -> list[Candidate]:
    """Fetch additional context from files imported by selected sources."""
    additional_candidates: list[Candidate] = []
    seen_files: set[str] = set(selected_sources)
    
    for source in selected_sources:
        imported_files = get_imported_files(conn, source)
        
        for imported_file in imported_files:
            if imported_file in seen_files:
                continue
            seen_files.add(imported_file)
            
            # Do a quick FTS search in the imported file
            rows = conn.execute(
                """
                SELECT c.id, c.text, c.chunk_index, c.token_estimate, d.source
                FROM chunks_fts f
                JOIN chunks c ON f.rowid = c.id
                JOIN documents d ON c.document_id = d.id
                WHERE chunks_fts MATCH ? AND d.source = ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, imported_file, limit_per_file),
            ).fetchall()
            
            for row in rows:
                additional_candidates.append(
                    Candidate(
                        chunk_id=int(row[0]),
                        text=str(row[1]),
                        chunk_index=int(row[2]),
                        token_estimate=int(row[3]),
                        source=str(row[4]),
                        fts_rank=len(additional_candidates),
                    )
                )
    
    return additional_candidates
_CODE_BOUNDARY_PATTERNS: dict[str, re.Pattern[str]] = {
    ".py": re.compile(
        r"^(?:class\s+\w|def\s+\w|async\s+def\s+\w|@\w+)", re.MULTILINE
    ),
    ".js": re.compile(
        r"^(?:(?:export\s+)?(?:(?:async\s+)?function\s+\w|class\s+\w|const\s+\w+\s*=\s*(?:async\s+)?\(|module\.exports))",
        re.MULTILINE,
    ),
    ".ts": re.compile(
        r"^(?:(?:export\s+)?(?:(?:async\s+)?function\s+\w|class\s+\w|interface\s+\w|type\s+\w|const\s+\w+\s*=\s*(?:async\s+)?\())",
        re.MULTILINE,
    ),
    ".tsx": re.compile(
        r"^(?:(?:export\s+)?(?:(?:async\s+)?function\s+\w|class\s+\w|interface\s+\w|type\s+\w|const\s+\w+\s*=))",
        re.MULTILINE,
    ),
    ".jsx": re.compile(
        r"^(?:(?:export\s+)?(?:(?:async\s+)?function\s+\w|class\s+\w|const\s+\w+\s*=))",
        re.MULTILINE,
    ),
    ".java": re.compile(
        r"^(?:\s*(?:public|private|protected|static|abstract|final)\s+.*(?:class\s+\w|interface\s+\w|\w+\s*\())",
        re.MULTILINE,
    ),
    ".kt": re.compile(
        r"^(?:\s*(?:fun\s+|class\s+|interface\s+|object\s+|data\s+class\s+))", re.MULTILINE
    ),
    ".go": re.compile(r"^(?:func\s+|type\s+\w+\s+(?:struct|interface))", re.MULTILINE),
    ".rs": re.compile(
        r"^(?:(?:pub\s+)?(?:fn\s+|struct\s+|enum\s+|impl\s+|trait\s+|mod\s+))", re.MULTILINE
    ),
    ".c": re.compile(
        r"^(?:\w[\w\s\*]+\s+\w+\s*\(|typedef\s+|struct\s+\w+)", re.MULTILINE
    ),
    ".h": re.compile(
        r"^(?:\w[\w\s\*]+\s+\w+\s*\(|typedef\s+|struct\s+\w+|#define\s+\w+)", re.MULTILINE
    ),
    ".cpp": re.compile(
        r"^(?:\w[\w\s\*:&]+\s+\w+\s*\(|class\s+\w+|namespace\s+\w+|template)", re.MULTILINE
    ),
    ".cs": re.compile(
        r"^(?:\s*(?:public|private|protected|internal|static|abstract)\s+.*(?:class\s+\w|interface\s+\w|\w+\s*\())",
        re.MULTILINE,
    ),
    ".php": re.compile(
        r"^(?:\s*(?:public|private|protected|static)?\s*function\s+\w|class\s+\w)", re.MULTILINE
    ),
    ".rb": re.compile(r"^(?:def\s+\w|class\s+\w|module\s+\w)", re.MULTILINE),
    ".swift": re.compile(
        r"^(?:\s*(?:func\s+|class\s+|struct\s+|enum\s+|protocol\s+|extension\s+))", re.MULTILINE
    ),
}

# Tree-sitter language mapping: extension -> (language_name, grammar_package_name)
_TREE_SITTER_LANGUAGES: dict[str, tuple[str, str]] = {
    ".py": ("python", "tree_sitter_python"),
    ".js": ("javascript", "tree_sitter_javascript"),
    ".ts": ("typescript", "tree_sitter_typescript"),
    ".tsx": ("tsx", "tree_sitter_typescript"),
    ".jsx": ("javascript", "tree_sitter_javascript"),
    ".java": ("java", "tree_sitter_java"),
    ".go": ("go", "tree_sitter_go"),
    ".rs": ("rust", "tree_sitter_rust"),
    ".c": ("c", "tree_sitter_c"),
    ".h": ("c", "tree_sitter_c"),
    ".cpp": ("cpp", "tree_sitter_cpp"),
    ".rb": ("ruby", "tree_sitter_ruby"),
}

# AST node types that represent top-level semantic boundaries
_AST_BOUNDARY_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "async_function_definition", "class_definition", "decorated_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition", "arrow_function", "export_statement"},
    "typescript": {"function_declaration", "class_declaration", "method_definition", "interface_declaration", "type_alias_declaration", "export_statement"},
    "tsx": {"function_declaration", "class_declaration", "method_definition", "interface_declaration", "type_alias_declaration", "export_statement"},
    "java": {"class_declaration", "interface_declaration", "method_declaration", "constructor_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item", "trait_item", "mod_item"},
    "c": {"function_definition", "struct_specifier", "type_definition"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier", "namespace_definition"},
    "ruby": {"method", "class", "module", "singleton_method"},
}

# Cached tree-sitter parsers
_TREE_SITTER_PARSER_CACHE: dict[str, object] = {}
_TREE_SITTER_AVAILABLE: bool | None = None


def _is_tree_sitter_available() -> bool:
    """Check if tree-sitter is available and working."""
    global _TREE_SITTER_AVAILABLE
    if _TREE_SITTER_AVAILABLE is not None:
        return _TREE_SITTER_AVAILABLE
    try:
        import tree_sitter  # type: ignore
        _TREE_SITTER_AVAILABLE = True
    except ImportError:
        _TREE_SITTER_AVAILABLE = False
    return _TREE_SITTER_AVAILABLE


def _get_tree_sitter_parser(ext: str) -> object | None:
    """Get or create a tree-sitter parser for the given file extension."""
    if ext not in _TREE_SITTER_LANGUAGES:
        return None
    if ext in _TREE_SITTER_PARSER_CACHE:
        return _TREE_SITTER_PARSER_CACHE[ext]

    lang_name, grammar_module = _TREE_SITTER_LANGUAGES[ext]

    try:
        import tree_sitter  # type: ignore
        import importlib
        grammar = importlib.import_module(grammar_module)

        # Handle typescript/tsx which have multiple languages in one package
        if lang_name == "typescript":
            language = tree_sitter.Language(grammar.language_typescript())
        elif lang_name == "tsx":
            language = tree_sitter.Language(grammar.language_tsx())
        else:
            language = tree_sitter.Language(grammar.language())

        parser = tree_sitter.Parser(language)
        _TREE_SITTER_PARSER_CACHE[ext] = parser
        return parser
    except Exception:
        return None


def _extract_ast_chunks(text: str, ext: str, chunk_size_words: int) -> list[str] | None:
    """Extract semantic chunks from code using tree-sitter AST parsing.

    Returns None if tree-sitter parsing fails or is unavailable, allowing
    fallback to regex-based chunking.
    """
    if not _is_tree_sitter_available():
        return None

    parser = _get_tree_sitter_parser(ext)
    if parser is None:
        return None

    lang_name = _TREE_SITTER_LANGUAGES.get(ext, (None, None))[0]
    if lang_name is None:
        return None

    boundary_types = _AST_BOUNDARY_NODE_TYPES.get(lang_name, set())
    if not boundary_types:
        return None

    try:
        tree = parser.parse(text.encode("utf-8"))  # type: ignore
    except Exception:
        return None

    # Collect top-level semantic nodes
    segments: list[str] = []

    def collect_nodes(node: object, depth: int = 0) -> None:
        """Recursively collect semantic boundary nodes."""
        node_type = getattr(node, "type", "")
        if node_type in boundary_types:
            start_byte = getattr(node, "start_byte", 0)
            end_byte = getattr(node, "end_byte", len(text))
            segment_text = text[start_byte:end_byte].strip()
            if segment_text:
                segments.append(segment_text)
            return  # Don't recurse into nested definitions to avoid duplication

        # For module/program root, recurse into children
        children = getattr(node, "children", [])
        for child in children:
            collect_nodes(child, depth + 1)

    root = getattr(tree, "root_node", None)
    if root is None:
        return None

    collect_nodes(root)

    if not segments:
        return None

    # Merge small segments or split oversized ones
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_words = 0
    max_words = max(chunk_size_words, MIN_CHUNK_WORDS)

    for segment in segments:
        seg_words = len(segment.split())
        if seg_words > max_words * 2:
            # Flush buffer first
            if buffer:
                chunks.append("\n\n".join(buffer))
                buffer, buffer_words = [], 0
            # Split oversized segment with plain chunker (no overlap for code)
            sub_chunks = chunk_text(segment, chunk_size_words, overlap_words=0)
            chunks.extend(sub_chunks)
        elif buffer_words + seg_words > max_words and buffer:
            chunks.append("\n\n".join(buffer))
            buffer, buffer_words = [segment], seg_words
        else:
            buffer.append(segment)
            buffer_words += seg_words

    if buffer:
        chunks.append("\n\n".join(buffer))

    return [c for c in chunks if c.strip()]

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "__pycache__",
    ".next",
    ".idea",
    ".vscode",
}

NOISE_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(r"^\s*Table of Contents\s*$", re.IGNORECASE),
    re.compile(r"^\s*Copyright\s+\(c\)\s+\d{4}", re.IGNORECASE),
    re.compile(r"^\s*All rights reserved\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Generated by\s+", re.IGNORECASE),
]


@dataclass
class Candidate:
    chunk_id: int
    source: str
    chunk_index: int
    text: str
    token_estimate: int
    bm25_score: float | None = None
    fts_rank: int | None = None
    vector_rank: int | None = None
    fts_score: float = 0.0
    vector_score: float = 0.0
    overlap_score: float = 0.0
    final_score: float = 0.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return cleaned[:120] if cleaned else "default"


def hash_text(value: str) -> str:
    return blake2b(value.encode("utf-8"), digest_size=16).hexdigest()


def embedding_cache_key(
    text: str,
    embedding_backend: str,
    embedding_model: str | None,
    dimensions: int,
) -> str:
    return "|".join(
        [
            embedding_backend,
            embedding_model or "",
            str(dimensions),
            hash_text(text),
        ]
    )


def ann_artifact_prefix(
    db_path: Path,
    backend: str,
    dimensions: int,
    model_name: str | None,
) -> Path:
    model_slug = safe_slug(model_name or "none")
    stem = f"{safe_slug(backend)}_{dimensions}_{model_slug}"
    return db_path.parent / "ann" / stem


def session_memory_path(db_path: Path) -> Path:
    return db_path.parent / "session_memory.json"


def load_session_memory(path: Path) -> dict:
    if not path.exists():
        return {"sessions": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"sessions": {}}


def save_session_memory(path: Path, memory: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory, ensure_ascii=False), encoding="utf-8")


def get_recent_session_queries(memory: dict, session_id: str, limit: int = 4) -> list[str]:
    sessions = memory.get("sessions", {})
    session = sessions.get(session_id, {})
    recent = session.get("recent_queries", [])
    if not isinstance(recent, list):
        return []
    return [str(x) for x in recent[-limit:]]


def update_session_memory(
    memory_path: Path,
    session_id: str,
    query: str,
    selected_sources: list[str],
) -> dict:
    memory = load_session_memory(memory_path)
    sessions = memory.setdefault("sessions", {})
    session = sessions.setdefault(session_id, {"recent_queries": [], "recent_sources": []})

    recent_queries = session.setdefault("recent_queries", [])
    recent_queries.append(query)
    session["recent_queries"] = recent_queries[-8:]

    recent_sources = session.setdefault("recent_sources", [])
    for src in selected_sources:
        if src not in recent_sources:
            recent_sources.append(src)
    session["recent_sources"] = recent_sources[-12:]

    session["updated_at"] = utc_now_iso()
    sessions[session_id] = session
    memory["sessions"] = sessions
    save_session_memory(memory_path, memory)
    return memory


def get_query_embedding(
    conn: sqlite3.Connection,
    query: str,
    dimensions: int,
    embedding_backend: str,
    embedding_model: str | None,
) -> tuple[list[float], str, str | None]:
    requested_key = embedding_cache_key(
        text=query,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        dimensions=dimensions,
    )
    in_memory = _EMBEDDING_VECTOR_CACHE.get(requested_key)
    if in_memory is not None:
        return in_memory, embedding_backend, embedding_model

    persisted = get_cached_query_embedding(conn=conn, query_key=requested_key)
    if persisted is not None:
        _EMBEDDING_VECTOR_CACHE[requested_key] = persisted
        return persisted, embedding_backend, embedding_model

    vec, effective_backend, effective_model = embed_text(
        text=query,
        dimensions=dimensions,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
    )

    effective_key = embedding_cache_key(
        text=query,
        embedding_backend=effective_backend,
        embedding_model=effective_model,
        dimensions=len(vec),
    )
    query_hash = hash_text(query)

    _EMBEDDING_VECTOR_CACHE[effective_key] = vec
    if effective_key != requested_key:
        _EMBEDDING_VECTOR_CACHE[requested_key] = vec

    set_cached_query_embedding(
        conn=conn,
        query_key=effective_key,
        query_text_hash=query_hash,
        embedding=vec,
        backend=effective_backend,
        model_name=effective_model,
    )
    if effective_key != requested_key:
        set_cached_query_embedding(
            conn=conn,
            query_key=requested_key,
            query_text_hash=query_hash,
            embedding=vec,
            backend=effective_backend,
            model_name=effective_model,
        )

    return vec, effective_backend, effective_model


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def char_ngrams(token: str, n: int = 3) -> list[str]:
    padded = f"^{token}$"
    if len(padded) < n:
        return [padded]
    return [padded[i : i + n] for i in range(0, len(padded) - n + 1)]


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def normalize_chunking(chunk_size_words: int, overlap_words: int) -> tuple[int, int]:
    normalized_chunk = chunk_size_words
    normalized_overlap = overlap_words

    if normalized_chunk < MIN_CHUNK_WORDS:
        print(
            f"[warn] chunk-size {normalized_chunk} too small; raised to {MIN_CHUNK_WORDS} words (~{estimate_tokens('x ' * MIN_CHUNK_WORDS)} token-est).",
            file=sys.stderr,
        )
        normalized_chunk = MIN_CHUNK_WORDS

    if normalized_chunk > MAX_CHUNK_WORDS:
        print(
            f"[warn] chunk-size {normalized_chunk} too large; capped at {MAX_CHUNK_WORDS} words (~{MAX_CHUNK_TOKEN_ESTIMATE} token-est).",
            file=sys.stderr,
        )
        normalized_chunk = MAX_CHUNK_WORDS

    max_overlap = max(20, int(normalized_chunk * 0.45))
    if normalized_overlap >= normalized_chunk:
        normalized_overlap = max(20, int(normalized_chunk * 0.2))
        print(
            f"[warn] overlap must be smaller than chunk-size; adjusted to {normalized_overlap} words.",
            file=sys.stderr,
        )
    elif normalized_overlap > max_overlap:
        print(
            f"[warn] overlap {normalized_overlap} too high; capped at {max_overlap} words.",
            file=sys.stderr,
        )
        normalized_overlap = max_overlap

    return normalized_chunk, normalized_overlap


def validate_query_input(query: str) -> tuple[bool, str | None]:
    word_count = len(query.split())
    line_count = query.count("\n") + 1

    if word_count > MAX_QUERY_WORDS or line_count > MAX_QUERY_LINES:
        return (
            False,
            (
                f"Query looks like pasted raw corpus ({word_count} words, {line_count} lines). "
                "Keep query concise and pass large files/logs via --inputs, then rerun."
            ),
        )

    return True, None


def clean_text(text: str) -> str:
    cleaned_lines: list[str] = []
    seen_recent: set[str] = set()

    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        if any(pattern.match(line) for pattern in NOISE_PATTERNS):
            continue
        if len(line) < 2:
            continue
        lowered = line.lower()
        if lowered in seen_recent:
            continue
        if len(seen_recent) > 5000:
            seen_recent.clear()
        seen_recent.add(lowered)
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def chunk_text(text: str, chunk_size_words: int, overlap_words: int) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[str] = []
    cursor = 0

    while cursor < len(lines):
        current: list[str] = []
        words_in_chunk = 0
        end = cursor

        while end < len(lines):
            line = lines[end]
            line_words = max(1, len(line.split()))
            if current and words_in_chunk + line_words > chunk_size_words:
                break
            current.append(line)
            words_in_chunk += line_words
            end += 1

        chunk = "\n".join(current).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(lines):
            break

        rewind_words = 0
        rewind_cursor = end - 1
        while rewind_cursor >= cursor and rewind_words < overlap_words:
            rewind_words += max(1, len(lines[rewind_cursor].split()))
            rewind_cursor -= 1

        next_cursor = max(cursor + 1, rewind_cursor + 1)
        cursor = next_cursor

    return chunks


def is_code_file(path: str) -> bool:
    return Path(path).suffix.lower() in CODE_EXTENSIONS


def chunk_code(text: str, source: str, chunk_size_words: int) -> list[str]:
    """Split code by function/class boundaries using AST parsing when available.

    Uses tree-sitter for true semantic AST chunking when available.
    Falls back to regex-based chunking, then to standard chunk_text() if
    no language pattern matches or if the file has no recognizable boundaries.
    """
    ext = Path(source).suffix.lower()

    # Try AST-based chunking first (when tree-sitter is available)
    if _is_tree_sitter_available() and ext in _TREE_SITTER_LANGUAGES:
        ast_chunks = _extract_ast_chunks(text, ext, chunk_size_words)
        if ast_chunks:
            return ast_chunks

    # Fall back to regex-based chunking
    pattern = _CODE_BOUNDARY_PATTERNS.get(ext)
    if pattern is None:
        return chunk_text(text, chunk_size_words, overlap_words=0)

    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    # Find line indices where a new top-level symbol starts
    boundary_lines: list[int] = []
    for i, line in enumerate(lines):
        if pattern.match(line):
            boundary_lines.append(i)

    if not boundary_lines:
        return chunk_text(text, chunk_size_words, overlap_words=0)

    # Build segments between boundaries
    segments: list[str] = []
    if boundary_lines[0] > 0:
        preamble = "".join(lines[: boundary_lines[0]]).strip()
        if preamble:
            segments.append(preamble)

    for idx, start in enumerate(boundary_lines):
        end = boundary_lines[idx + 1] if idx + 1 < len(boundary_lines) else len(lines)
        segment = "".join(lines[start:end]).strip()
        if segment:
            segments.append(segment)

    # Merge small segments or split oversized ones
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_words = 0
    max_words = max(chunk_size_words, MIN_CHUNK_WORDS)

    for segment in segments:
        seg_words = len(segment.split())
        if seg_words > max_words * 2:
            # Flush buffer first
            if buffer:
                chunks.append("\n\n".join(buffer))
                buffer, buffer_words = [], 0
            # Split oversized segment with plain chunker (no overlap for code)
            sub_chunks = chunk_text(segment, chunk_size_words, overlap_words=0)
            chunks.extend(sub_chunks)
        elif buffer_words + seg_words > max_words and buffer:
            chunks.append("\n\n".join(buffer))
            buffer, buffer_words = [segment], seg_words
        else:
            buffer.append(segment)
            buffer_words += seg_words

    if buffer:
        chunks.append("\n\n".join(buffer))

    return [c for c in chunks if c.strip()]


def embed_text_hash(text: str, dimensions: int) -> list[float]:
    """Create hash-based embeddings using locality-sensitive hashing.

    Note: Hash embeddings provide lexical similarity similar to FTS5/BM25.
    When hashEmbeddingSkipVector is true (default), vector retrieval is
    skipped entirely when using hash backend to avoid redundant noise.
    """
    vec = [0.0] * dimensions
    terms = tokenize(text)
    if not terms:
        return vec

    ngram_weight = get_weight("char_ngram_weight")

    term_counts: dict[str, float] = {}
    for term in terms:
        term_counts[f"w:{term}"] = term_counts.get(f"w:{term}", 0.0) + 1.0
        for gram in char_ngrams(term):
            key = f"g:{gram}"
            term_counts[key] = term_counts.get(key, 0.0) + ngram_weight

    for term, count in term_counts.items():
        digest = blake2b(term.encode("utf-8"), digest_size=16).digest()
        idx = int.from_bytes(digest[:8], byteorder="big") % dimensions
        sign = 1.0 if digest[8] % 2 == 0 else -1.0
        weight = 1.0 + math.log1p(count)
        vec[idx] += sign * weight

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def get_sentence_transformer_model(model_name: str):
    cached = _EMBEDDING_MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "ML embedding backend requested but sentence-transformers is not installed."
        ) from exc

    model = SentenceTransformer(model_name)
    _EMBEDDING_MODEL_CACHE[model_name] = model
    return model


def embed_text_ml(text: str, model_name: str) -> list[float]:
    model = get_sentence_transformer_model(model_name)
    embedding = model.encode([text], normalize_embeddings=True)[0]
    return [float(x) for x in embedding]


def resolve_embedding_backend(
    requested_backend: str,
    requested_model: str,
) -> tuple[str, str | None]:
    backend = requested_backend.strip().lower()
    if backend == "hash":
        return "hash", None

    if backend == "ml":
        try:
            get_sentence_transformer_model(requested_model)
            return "ml", requested_model
        except Exception as exc:
            print(
                f"[warn] ML embedding backend unavailable ({exc}). Falling back to hash embeddings.",
                file=sys.stderr,
            )
            return "hash", None

    print(
        f"[warn] Unknown embedding backend '{requested_backend}'. Falling back to hash.",
        file=sys.stderr,
    )
    return "hash", None


def embed_text(
    text: str,
    dimensions: int,
    embedding_backend: str,
    embedding_model: str | None,
) -> tuple[list[float], str, str | None]:
    backend = embedding_backend.strip().lower()
    if backend == "ml" and embedding_model:
        try:
            return embed_text_ml(text=text, model_name=embedding_model), "ml", embedding_model
        except Exception as exc:
            print(
                f"[warn] ML embedding runtime failed ({exc}). Falling back to hash embeddings.",
                file=sys.stderr,
            )

    return embed_text_hash(text=text, dimensions=dimensions), "hash", None


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return float(sum(a * b for a, b in zip(vec_a, vec_b)))


def rank_score(rank: int | None) -> float:
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / float(rank)


def bm25_to_score(bm25_value: float | None) -> float:
    if bm25_value is None:
        return 0.0
    return 1.0 / (1.0 + abs(float(bm25_value)))


MAX_FILE_SIZE_BYTES = 512 * 1024  # 512 KB — skip files larger than this

_MINIFIED_PATTERN = re.compile(r"[^\n]{500,}")  # lines >500 chars suggest minification


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS


def should_skip_file(path: Path) -> bool:
    """Return True for autogenerated, minified, or token-dense files that add noise."""
    name = path.name.lower()
    if name in IGNORED_FILENAMES:
        return True
    # Block common autogenerated/bundled patterns
    if name.endswith(".min.js") or name.endswith(".min.css"):
        return True
    if name.endswith(".bundle.js") or name.endswith(".chunk.js"):
        return True
    if name.endswith(".map"):  # source maps
        return True
    # SVG files often contain massive path data
    if path.suffix.lower() == ".svg":
        return True
    # Skip oversized files
    try:
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            print(f"[skip] File too large ({path.stat().st_size} bytes): {path.name}", file=sys.stderr)
            return True
    except OSError:
        return True
    return False


def is_minified(text: str, sample_lines: int = 10) -> bool:
    """Detect minified content by checking for extremely long lines."""
    for i, line in enumerate(text.splitlines()):
        if i >= sample_lines:
            break
        if len(line) > 500:
            return True
    return False


def collect_input_files(inputs: Sequence[str]) -> list[Path]:
    files: set[Path] = set()

    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            print(f"[warn] Path does not exist: {path}", file=sys.stderr)
            continue

        if path.is_file():
            if is_text_file(path) and not should_skip_file(path):
                files.add(path)
            continue

        for root, dirs, names in os.walk(path):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            for name in names:
                candidate = Path(root) / name
                if is_text_file(candidate) and not should_skip_file(candidate):
                    files.add(candidate.resolve())

    return sorted(files)


def read_text_file(path: Path) -> str | None:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL UNIQUE,
                raw_text TEXT NOT NULL,
                cleaned_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                token_estimate INTEGER NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE(document_id, chunk_index)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(document_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(text, content='chunks', content_rowid='id');

            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id INTEGER PRIMARY KEY,
                embedding_json TEXT NOT NULL,
                backend TEXT NOT NULL DEFAULT 'hash',
                dimensions INTEGER NOT NULL DEFAULT 256,
                model_name TEXT,
                FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_backend_dims
            ON chunk_embeddings(backend, dimensions);

            CREATE TABLE IF NOT EXISTS query_embeddings (
                query_key TEXT PRIMARY KEY,
                query_text_hash TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                backend TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                model_name TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_query_embeddings_backend_dims
            ON query_embeddings(backend, dimensions);

            CREATE TABLE IF NOT EXISTS query_cache (
                cache_key TEXT PRIMARY KEY,
                cache_payload_json TEXT NOT NULL,
                created_at_epoch INTEGER NOT NULL,
                expires_at_epoch INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_query_cache_expires
            ON query_cache(expires_at_epoch);

            -- Import graph for "fake LSP" functionality
            CREATE TABLE IF NOT EXISTS file_dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL,
                target_import TEXT NOT NULL,
                resolved_file TEXT,
                UNIQUE(source_file, target_import)
            );

            CREATE INDEX IF NOT EXISTS idx_file_deps_source
            ON file_dependencies(source_file);

            CREATE INDEX IF NOT EXISTS idx_file_deps_resolved
            ON file_dependencies(resolved_file);

            -- Symbol index for 2-hop expansion
            CREATE TABLE IF NOT EXISTS symbol_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                symbol_type TEXT,
                chunk_id INTEGER,
                line_start INTEGER,
                line_end INTEGER,
                signature TEXT,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id)
            );

            CREATE INDEX IF NOT EXISTS idx_symbol_name
            ON symbol_index(symbol_name);

            CREATE INDEX IF NOT EXISTS idx_symbol_file
            ON symbol_index(file_path);
            """
        )

        for statement in (
            "ALTER TABLE chunk_embeddings ADD COLUMN backend TEXT NOT NULL DEFAULT 'hash'",
            "ALTER TABLE chunk_embeddings ADD COLUMN dimensions INTEGER NOT NULL DEFAULT 256",
            "ALTER TABLE chunk_embeddings ADD COLUMN model_name TEXT",
        ):
            try:
                conn.execute(statement)
            except sqlite3.OperationalError:
                pass

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_backend_dims ON chunk_embeddings(backend, dimensions);"
        )
    except sqlite3.OperationalError as exc:
        if "fts5" in str(exc).lower():
            raise RuntimeError("SQLite FTS5 is not available in this Python build.") from exc
        raise

    return conn


def get_index_fingerprint(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        """
        SELECT COUNT(*) AS chunk_count, COALESCE(MAX(updated_at), '') AS max_updated
        FROM documents
        """
    ).fetchone()
    chunk_row = conn.execute("SELECT COUNT(*) AS total FROM chunks").fetchone()
    chunk_count = int(chunk_row["total"]) if chunk_row else 0
    doc_count = int(row["chunk_count"]) if row and row["chunk_count"] is not None else 0
    max_updated = str(row["max_updated"]) if row and row["max_updated"] is not None else ""
    return hash_text(f"docs={doc_count}|chunks={chunk_count}|updated={max_updated}")


def cleanup_query_cache(conn: sqlite3.Connection, now_epoch: int) -> None:
    conn.execute("DELETE FROM query_cache WHERE expires_at_epoch <= ?", (now_epoch,))
    conn.commit()


def get_cached_query_result(conn: sqlite3.Connection, cache_key: str, now_epoch: int) -> dict | None:
    row = conn.execute(
        """
        SELECT cache_payload_json, expires_at_epoch
        FROM query_cache
        WHERE cache_key = ?
        """,
        (cache_key,),
    ).fetchone()
    if not row:
        return None

    if int(row["expires_at_epoch"]) <= now_epoch:
        conn.execute("DELETE FROM query_cache WHERE cache_key = ?", (cache_key,))
        conn.commit()
        return None

    try:
        return json.loads(str(row["cache_payload_json"]))
    except Exception:
        return None


def set_cached_query_result(
    conn: sqlite3.Connection,
    cache_key: str,
    payload: dict,
    now_epoch: int,
    ttl_seconds: int,
) -> None:
    expires_at = now_epoch + max(30, ttl_seconds)
    conn.execute(
        """
        INSERT INTO query_cache (cache_key, cache_payload_json, created_at_epoch, expires_at_epoch)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
            cache_payload_json = excluded.cache_payload_json,
            created_at_epoch = excluded.created_at_epoch,
            expires_at_epoch = excluded.expires_at_epoch
        """,
        (cache_key, json.dumps(payload, separators=(",", ":")), now_epoch, expires_at),
    )
    conn.commit()


def get_cached_query_embedding(conn: sqlite3.Connection, query_key: str) -> list[float] | None:
    row = conn.execute(
        "SELECT embedding_json FROM query_embeddings WHERE query_key = ?",
        (query_key,),
    ).fetchone()
    if not row:
        return None
    try:
        return [float(x) for x in json.loads(str(row["embedding_json"]))]
    except Exception:
        return None


def set_cached_query_embedding(
    conn: sqlite3.Connection,
    query_key: str,
    query_text_hash: str,
    embedding: list[float],
    backend: str,
    model_name: str | None,
) -> None:
    now_iso = utc_now_iso()
    conn.execute(
        """
        INSERT INTO query_embeddings (
            query_key,
            query_text_hash,
            embedding_json,
            backend,
            dimensions,
            model_name,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(query_key) DO UPDATE SET
            embedding_json = excluded.embedding_json,
            backend = excluded.backend,
            dimensions = excluded.dimensions,
            model_name = excluded.model_name,
            updated_at = excluded.updated_at
        """,
        (
            query_key,
            query_text_hash,
            json.dumps(embedding, separators=(",", ":")),
            backend,
            len(embedding),
            model_name,
            now_iso,
            now_iso,
        ),
    )
    conn.commit()


def upsert_document(
    conn: sqlite3.Connection,
    source: str,
    raw_text: str,
    cleaned_text: str,
    chunk_size_words: int,
    overlap_words: int,
    dimensions: int,
    embedding_backend: str,
    embedding_model: str | None,
) -> int:
    now = utc_now_iso()
    existing = conn.execute("SELECT id FROM documents WHERE source = ?", (source,)).fetchone()

    if existing:
        document_id = int(existing["id"])
        old_chunks = conn.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,)).fetchall()
        for row in old_chunks:
            cid = int(row["id"])
            conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (cid,))
            conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

        conn.execute(
            """
            UPDATE documents
            SET raw_text = ?, cleaned_text = ?, updated_at = ?
            WHERE id = ?
            """,
            (raw_text, cleaned_text, now, document_id),
        )
    else:
        cursor = conn.execute(
            """
            INSERT INTO documents (source, raw_text, cleaned_text, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source, raw_text, cleaned_text, now, now),
        )
        document_id = int(cursor.lastrowid)

    if is_code_file(source):
        chunks = chunk_code(cleaned_text, source, chunk_size_words)
    else:
        chunks = chunk_text(cleaned_text, chunk_size_words, overlap_words)
    for idx, chunk in enumerate(chunks):
        cursor = conn.execute(
            """
            INSERT INTO chunks (document_id, chunk_index, text, token_estimate)
            VALUES (?, ?, ?, ?)
            """,
            (document_id, idx, chunk, estimate_tokens(chunk)),
        )
        chunk_id = int(cursor.lastrowid)
        conn.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", (chunk_id, chunk))
        emb, effective_backend, effective_model = embed_text(
            text=chunk,
            dimensions=dimensions,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
        )
        conn.execute(
            """
            INSERT INTO chunk_embeddings (chunk_id, embedding_json, backend, dimensions, model_name)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                json.dumps(emb, separators=(",", ":")),
                effective_backend,
                len(emb),
                effective_model,
            ),
        )

    return len(chunks)


def index_corpus(
    conn: sqlite3.Connection,
    file_paths: Iterable[Path],
    chunk_size_words: int,
    overlap_words: int,
    dimensions: int,
    embedding_backend: str,
    embedding_model: str | None,
) -> dict[str, int]:
    files_indexed = 0
    chunks_indexed = 0

    for path in file_paths:
        raw = read_text_file(path)
        if raw is None:
            print(f"[warn] Could not read file: {path}", file=sys.stderr)
            continue

        if is_minified(raw):
            print(f"[skip] Minified content detected: {path.name}", file=sys.stderr)
            continue

        cleaned = clean_text(raw)
        if not cleaned:
            continue

        chunk_count = upsert_document(
            conn=conn,
            source=str(path),
            raw_text=raw,
            cleaned_text=cleaned,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            dimensions=dimensions,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
        )
        files_indexed += 1
        chunks_indexed += chunk_count

    conn.commit()
    return {"files_indexed": files_indexed, "chunks_indexed": chunks_indexed}


def try_import_hnsw_modules():
    try:
        import hnswlib  # type: ignore
        import numpy as np  # type: ignore

        return hnswlib, np
    except Exception:
        return None, None


def fetch_embeddings_for_ann(
    conn: sqlite3.Connection,
    backend: str,
    dimensions: int,
    model_name: str | None,
) -> list[sqlite3.Row]:
    sql = """
    SELECT chunk_id, embedding_json, dimensions, backend, model_name
    FROM chunk_embeddings
    WHERE backend = ? AND dimensions = ?
    """
    params: list[object] = [backend, dimensions]
    if backend == "ml" and model_name:
        sql += " AND (model_name = ? OR model_name IS NULL)"
        params.append(model_name)
    return conn.execute(sql, tuple(params)).fetchall()


def infer_embedding_dimensions(
    conn: sqlite3.Connection,
    backend: str,
    model_name: str | None,
    fallback_dimensions: int,
) -> int:
    sql = "SELECT dimensions FROM chunk_embeddings WHERE backend = ?"
    params: list[object] = [backend]
    if backend == "ml" and model_name:
        sql += " AND (model_name = ? OR model_name IS NULL)"
        params.append(model_name)
    sql += " ORDER BY rowid DESC LIMIT 1"
    row = conn.execute(sql, tuple(params)).fetchone()
    if row and row["dimensions"] is not None:
        return int(row["dimensions"])
    return fallback_dimensions


def build_hnsw_index(
    conn: sqlite3.Connection,
    db_path: Path,
    backend: str,
    dimensions: int,
    model_name: str | None,
) -> bool:
    hnswlib, np = try_import_hnsw_modules()
    if hnswlib is None or np is None:
        return False

    rows = fetch_embeddings_for_ann(
        conn=conn,
        backend=backend,
        dimensions=dimensions,
        model_name=model_name,
    )
    if not rows:
        return False

    ids: list[int] = []
    vectors: list[list[float]] = []
    for row in rows:
        try:
            vec = [float(x) for x in json.loads(str(row["embedding_json"]))]
        except Exception:
            continue
        if len(vec) != dimensions:
            continue
        ids.append(int(row["chunk_id"]))
        vectors.append(vec)

    if not ids:
        return False

    prefix = ann_artifact_prefix(
        db_path=db_path,
        backend=backend,
        dimensions=dimensions,
        model_name=model_name,
    )
    index_path = prefix.with_suffix(".hnsw.bin")
    meta_path = prefix.with_suffix(".hnsw.meta.json")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    idx = hnswlib.Index(space="cosine", dim=dimensions)
    idx.init_index(max_elements=len(ids), ef_construction=220, M=32)
    idx.add_items(np.array(vectors, dtype=np.float32), np.array(ids, dtype=np.int64))
    idx.set_ef(min(320, max(DEFAULT_ANN_EF_SEARCH, len(ids))))
    idx.save_index(str(index_path))

    meta_payload = {
        "engine": DEFAULT_ANN_ENGINE,
        "backend": backend,
        "dimensions": dimensions,
        "model_name": model_name,
        "elements": len(ids),
        "index_fingerprint": get_index_fingerprint(conn),
        "created_at": utc_now_iso(),
    }
    meta_path.write_text(json.dumps(meta_payload), encoding="utf-8")
    return True


def query_hnsw_index(
    conn: sqlite3.Connection,
    db_path: Path,
    query_vec: list[float],
    limit: int,
    backend: str,
    model_name: str | None,
) -> list[tuple[int, float]]:
    hnswlib, np = try_import_hnsw_modules()
    if hnswlib is None or np is None:
        return []

    dimensions = len(query_vec)
    prefix = ann_artifact_prefix(
        db_path=db_path,
        backend=backend,
        dimensions=dimensions,
        model_name=model_name,
    )
    index_path = prefix.with_suffix(".hnsw.bin")
    meta_path = prefix.with_suffix(".hnsw.meta.json")
    if not index_path.exists() or not meta_path.exists():
        built = build_hnsw_index(
            conn=conn,
            db_path=db_path,
            backend=backend,
            dimensions=dimensions,
            model_name=model_name,
        )
        if not built:
            return []

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    current_fp = get_index_fingerprint(conn)
    if meta.get("index_fingerprint") != current_fp:
        rebuilt = build_hnsw_index(
            conn=conn,
            db_path=db_path,
            backend=backend,
            dimensions=dimensions,
            model_name=model_name,
        )
        if not rebuilt:
            return []

    idx = hnswlib.Index(space="cosine", dim=dimensions)
    idx.load_index(str(index_path))
    idx.set_ef(DEFAULT_ANN_EF_SEARCH)

    current_count = int(idx.get_current_count())
    if current_count <= 0:
        return []

    k = min(max(1, limit), current_count)
    labels, distances = idx.knn_query(np.array([query_vec], dtype=np.float32), k=k)
    results: list[tuple[int, float]] = []
    for label, distance in zip(labels[0], distances[0]):
        if int(label) < 0:
            continue
        sim = max(0.0, 1.0 - float(distance))
        results.append((int(label), sim))
    return results


def try_import_faiss():
    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
        return faiss, np
    except Exception:
        return None, None


def build_faiss_index(
    conn: sqlite3.Connection,
    db_path: Path,
    backend: str,
    dimensions: int,
    model_name: str | None,
) -> bool:
    faiss, np = try_import_faiss()
    if faiss is None or np is None:
        return False

    rows = fetch_embeddings_for_ann(
        conn=conn, backend=backend, dimensions=dimensions, model_name=model_name,
    )
    if not rows:
        return False

    ids: list[int] = []
    vectors: list[list[float]] = []
    for row in rows:
        try:
            vec = [float(x) for x in json.loads(str(row["embedding_json"]))]
        except Exception:
            continue
        if len(vec) != dimensions:
            continue
        ids.append(int(row["chunk_id"]))
        vectors.append(vec)

    if not ids:
        return False

    prefix = ann_artifact_prefix(
        db_path=db_path, backend=backend, dimensions=dimensions, model_name=model_name,
    )
    index_path = prefix.with_suffix(".faiss.bin")
    meta_path = prefix.with_suffix(".faiss.meta.json")
    idmap_path = prefix.with_suffix(".faiss.idmap.json")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(data)

    n_elements = len(ids)
    if n_elements < 1000:
        index = faiss.IndexFlatIP(dimensions)
    else:
        n_clusters = min(int(math.sqrt(n_elements)), 256)
        quantizer = faiss.IndexFlatIP(dimensions)
        index = faiss.IndexIVFFlat(quantizer, dimensions, n_clusters, faiss.METRIC_INNER_PRODUCT)
        index.train(data)
        index.nprobe = min(16, n_clusters)

    index.add(data)
    faiss.write_index(index, str(index_path))

    idmap_path.write_text(json.dumps(ids, separators=(",", ":")), encoding="utf-8")
    meta_payload = {
        "engine": "faiss",
        "backend": backend,
        "dimensions": dimensions,
        "model_name": model_name,
        "elements": n_elements,
        "index_fingerprint": get_index_fingerprint(conn),
        "created_at": utc_now_iso(),
    }
    meta_path.write_text(json.dumps(meta_payload), encoding="utf-8")
    return True


def query_faiss_index(
    conn: sqlite3.Connection,
    db_path: Path,
    query_vec: list[float],
    limit: int,
    backend: str,
    model_name: str | None,
) -> list[tuple[int, float]]:
    faiss, np = try_import_faiss()
    if faiss is None or np is None:
        return []

    dimensions = len(query_vec)
    prefix = ann_artifact_prefix(
        db_path=db_path, backend=backend, dimensions=dimensions, model_name=model_name,
    )
    index_path = prefix.with_suffix(".faiss.bin")
    meta_path = prefix.with_suffix(".faiss.meta.json")
    idmap_path = prefix.with_suffix(".faiss.idmap.json")

    if not index_path.exists() or not meta_path.exists():
        built = build_faiss_index(
            conn=conn, db_path=db_path, backend=backend,
            dimensions=dimensions, model_name=model_name,
        )
        if not built:
            return []

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    current_fp = get_index_fingerprint(conn)
    if meta.get("index_fingerprint") != current_fp:
        rebuilt = build_faiss_index(
            conn=conn, db_path=db_path, backend=backend,
            dimensions=dimensions, model_name=model_name,
        )
        if not rebuilt:
            return []

    try:
        id_map = json.loads(idmap_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    index = faiss.read_index(str(index_path))
    qvec = np.array([query_vec], dtype=np.float32)
    faiss.normalize_L2(qvec)

    k = min(max(1, limit), index.ntotal)
    scores, indices = index.search(qvec, k)

    results: list[tuple[int, float]] = []
    for idx_pos, score in zip(indices[0], scores[0]):
        idx_pos = int(idx_pos)
        if idx_pos < 0 or idx_pos >= len(id_map):
            continue
        chunk_id = id_map[idx_pos]
        sim = max(0.0, float(score))
        results.append((chunk_id, sim))
    return results


def build_fts_query(query: str) -> str:
    terms = tokenize(query)[:20]
    if not terms:
        return query.strip()
    return " OR ".join(f'"{term}"' for term in terms)


def fts_retrieve(conn: sqlite3.Connection, query: str, limit: int) -> list[Candidate]:
    fts_query = build_fts_query(query)
    if not fts_query:
        return []

    try:
        rows = conn.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.text AS text,
                c.chunk_index AS chunk_index,
                c.token_estimate AS token_estimate,
                d.source AS source,
                bm25(chunks_fts) AS bm25_score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.id = c.document_id
            WHERE chunks_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        safe = " ".join(tokenize(query)[:20])
        if not safe:
            return []
        rows = conn.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.text AS text,
                c.chunk_index AS chunk_index,
                c.token_estimate AS token_estimate,
                d.source AS source,
                bm25(chunks_fts) AS bm25_score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.id = c.document_id
            WHERE chunks_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
            """,
            (safe, limit),
        ).fetchall()

    results: list[Candidate] = []
    for i, row in enumerate(rows, start=1):
        results.append(
            Candidate(
                chunk_id=int(row["chunk_id"]),
                source=str(row["source"]),
                chunk_index=int(row["chunk_index"]),
                text=str(row["text"]),
                token_estimate=int(row["token_estimate"]),
                bm25_score=float(row["bm25_score"]) if row["bm25_score"] is not None else None,
                fts_rank=i,
            )
        )
    return results


def vector_retrieve(
    conn: sqlite3.Connection,
    db_path: Path,
    query: str,
    limit: int,
    dimensions: int,
    embedding_backend: str,
    embedding_model: str | None,
) -> tuple[list[Candidate], str, str | None, str]:
    q_vec, effective_backend, effective_model = get_query_embedding(
        conn=conn,
        query=query,
        dimensions=dimensions,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
    )
    effective_dimensions = len(q_vec)

    def fetch_rows(backend: str, vector_dims: int, model_name: str | None):
        base_sql = """
        SELECT
            c.id AS chunk_id,
            c.text AS text,
            c.chunk_index AS chunk_index,
            c.token_estimate AS token_estimate,
            d.source AS source,
            ce.embedding_json AS embedding_json,
            ce.backend AS backend,
            ce.dimensions AS dimensions,
            ce.model_name AS model_name
        FROM chunk_embeddings ce
        JOIN chunks c ON c.id = ce.chunk_id
        JOIN documents d ON d.id = c.document_id
        WHERE ce.backend = ? AND ce.dimensions = ?
        """
        params: list[object] = [backend, vector_dims]
        if backend == "ml" and model_name:
            base_sql += " AND (ce.model_name = ? OR ce.model_name IS NULL)"
            params.append(model_name)
        return conn.execute(base_sql, tuple(params)).fetchall()

    ann_results = query_hnsw_index(
        conn=conn,
        db_path=db_path,
        query_vec=q_vec,
        limit=limit,
        backend=effective_backend,
        model_name=effective_model,
    )

    if ann_results:
        chunk_ids = [cid for cid, _ in ann_results]
        placeholder = ",".join("?" for _ in chunk_ids)
        detail_rows = conn.execute(
            f"""
            SELECT
                c.id AS chunk_id,
                c.text AS text,
                c.chunk_index AS chunk_index,
                c.token_estimate AS token_estimate,
                d.source AS source
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholder})
            """,
            tuple(chunk_ids),
        ).fetchall()
        detail_map = {int(r["chunk_id"]): r for r in detail_rows}

        top: list[Candidate] = []
        for i, (chunk_id, sim) in enumerate(ann_results, start=1):
            row = detail_map.get(chunk_id)
            if row is None:
                continue
            top.append(
                Candidate(
                    chunk_id=int(row["chunk_id"]),
                    source=str(row["source"]),
                    chunk_index=int(row["chunk_index"]),
                    text=str(row["text"]),
                    token_estimate=int(row["token_estimate"]),
                    vector_score=float(sim),
                    vector_rank=i,
                )
            )
        if top:
            return top, effective_backend, effective_model, "hnsw"

    # Try FAISS as second ANN engine if HNSW didn't produce results
    faiss_results = query_faiss_index(
        conn=conn,
        db_path=db_path,
        query_vec=q_vec,
        limit=limit,
        backend=effective_backend,
        model_name=effective_model,
    )
    if faiss_results:
        chunk_ids = [cid for cid, _ in faiss_results]
        placeholder = ",".join("?" for _ in chunk_ids)
        detail_rows = conn.execute(
            f"""
            SELECT
                c.id AS chunk_id,
                c.text AS text,
                c.chunk_index AS chunk_index,
                c.token_estimate AS token_estimate,
                d.source AS source
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholder})
            """,
            tuple(chunk_ids),
        ).fetchall()
        detail_map = {int(r["chunk_id"]): r for r in detail_rows}
        top: list[Candidate] = []
        for i, (chunk_id, sim) in enumerate(faiss_results, start=1):
            row = detail_map.get(chunk_id)
            if row is None:
                continue
            top.append(
                Candidate(
                    chunk_id=int(row["chunk_id"]),
                    source=str(row["source"]),
                    chunk_index=int(row["chunk_index"]),
                    text=str(row["text"]),
                    token_estimate=int(row["token_estimate"]),
                    vector_score=float(sim),
                    vector_rank=i,
                )
            )
        if top:
            return top, effective_backend, effective_model, "faiss"

    rows = fetch_rows(effective_backend, effective_dimensions, effective_model)

    if not rows and effective_backend == "ml":
        print(
            "[warn] No ML embeddings found in index for selected model; falling back to hash vector retrieval.",
            file=sys.stderr,
        )
        q_vec, effective_backend, effective_model = get_query_embedding(
            conn=conn,
            query=query,
            dimensions=dimensions,
            embedding_backend="hash",
            embedding_model=None,
        )
        effective_dimensions = len(q_vec)
        rows = fetch_rows(effective_backend, effective_dimensions, effective_model)

    scored: list[tuple[float, Candidate]] = []
    for row in rows:
        try:
            emb = json.loads(str(row["embedding_json"]))
        except json.JSONDecodeError:
            continue

        sim = cosine_similarity(q_vec, emb)
        if sim <= 0:
            continue

        scored.append(
            (
                sim,
                Candidate(
                    chunk_id=int(row["chunk_id"]),
                    source=str(row["source"]),
                    chunk_index=int(row["chunk_index"]),
                    text=str(row["text"]),
                    token_estimate=int(row["token_estimate"]),
                    vector_score=sim,
                ),
            )
        )

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = [candidate for _, candidate in scored[:limit]]
    for i, candidate in enumerate(top, start=1):
        candidate.vector_rank = i
    return top, effective_backend, effective_model, "scan"


def overlap_ratio(query: str, text: str) -> float:
    q_terms = set(tokenize(query))
    if not q_terms:
        return 0.0
    t_terms = set(tokenize(text))
    return len(q_terms & t_terms) / float(len(q_terms))


def rerank_candidates(
    query: str,
    fts_hits: list[Candidate],
    vector_hits: list[Candidate],
    top_k: int,
) -> tuple[list[Candidate], list[Candidate]]:
    merged: dict[int, Candidate] = {}

    for candidate in fts_hits:
        merged[candidate.chunk_id] = candidate

    for candidate in vector_hits:
        existing = merged.get(candidate.chunk_id)
        if existing is None:
            merged[candidate.chunk_id] = candidate
        else:
            existing.vector_rank = candidate.vector_rank
            existing.vector_score = candidate.vector_score

    # Get configurable weights
    fts_lexical_w = get_weight("fts_lexical_rank_weight")
    fts_bm25_w = get_weight("fts_bm25_weight")
    final_fts_w = get_weight("final_fts_weight")
    final_vector_w = get_weight("final_vector_weight")
    final_overlap_w = get_weight("final_overlap_weight")

    ranked = list(merged.values())
    for item in ranked:
        lexical_rank_signal = rank_score(item.fts_rank)
        bm25_signal = bm25_to_score(item.bm25_score)
        item.fts_score = (fts_lexical_w * lexical_rank_signal) + (fts_bm25_w * bm25_signal)
        item.vector_score = item.vector_score if item.vector_score > 0 else rank_score(item.vector_rank)
        item.overlap_score = overlap_ratio(query, item.text)
        item.final_score = (final_fts_w * item.fts_score) + (final_vector_w * item.vector_score) + (final_overlap_w * item.overlap_score)

    ranked.sort(key=lambda c: c.final_score, reverse=True)
    bounded_top_k = max(3, min(top_k, 5))
    return ranked[:bounded_top_k], ranked


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def textrank_score_sentences(sentences: list[str], damping: float = 0.85, iterations: int = 30) -> list[tuple[float, str]]:
    """Score sentences using TextRank algorithm for extractive summarization.
    
    TextRank builds a graph where sentences are nodes and edges are weighted
    by semantic similarity. Sentences that are similar to many other important
    sentences receive higher scores - capturing semantic centrality.
    """
    if len(sentences) <= 2:
        return [(1.0, s) for s in sentences]
    
    n = len(sentences)
    
    # Build similarity matrix using Jaccard similarity of word sets
    similarity_matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    sentence_words = [set(tokenize(s)) for s in sentences]
    
    for i in range(n):
        for j in range(i + 1, n):
            if not sentence_words[i] or not sentence_words[j]:
                sim = 0.0
            else:
                intersection = len(sentence_words[i] & sentence_words[j])
                union = len(sentence_words[i] | sentence_words[j])
                sim = intersection / union if union > 0 else 0.0
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim
    
    # Normalize rows (outgoing edge weights)
    for i in range(n):
        row_sum = sum(similarity_matrix[i])
        if row_sum > 0:
            for j in range(n):
                similarity_matrix[i][j] /= row_sum
    
    # Initialize scores uniformly
    scores = [1.0 / n] * n
    
    # Power iteration
    for _ in range(iterations):
        new_scores = [0.0] * n
        for i in range(n):
            rank_sum = sum(similarity_matrix[j][i] * scores[j] for j in range(n))
            new_scores[i] = (1 - damping) / n + damping * rank_sum
        scores = new_scores
    
    # Normalize final scores
    max_score = max(scores) if scores else 1.0
    if max_score > 0:
        scores = [s / max_score for s in scores]
    
    return list(zip(scores, sentences))


def cluster_chunks_semantically(
    chunks: list[str],
    embeddings: list[list[float]],
    num_clusters: int = 5,
) -> list[list[int]]:
    """Cluster chunks by semantic similarity using k-means on embeddings.
    
    Returns list of clusters, each containing chunk indices.
    This enables selecting diverse, representative chunks.
    """
    if len(chunks) <= num_clusters:
        return [[i] for i in range(len(chunks))]
    
    try:
        import numpy as np
    except ImportError:
        # Fallback: return all chunks as single cluster
        return [list(range(len(chunks)))]
    
    embeddings_arr = np.array(embeddings)
    n = len(chunks)
    k = min(num_clusters, n)
    
    # Simple k-means clustering
    # Initialize centroids randomly
    rng = np.random.default_rng(42)
    centroid_indices = rng.choice(n, size=k, replace=False)
    centroids = embeddings_arr[centroid_indices].copy()
    
    assignments = np.zeros(n, dtype=int)
    
    for _ in range(20):  # Max iterations
        # Assign points to nearest centroid
        for i in range(n):
            distances = [np.linalg.norm(embeddings_arr[i] - centroids[j]) for j in range(k)]
            assignments[i] = int(np.argmin(distances))
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            cluster_points = embeddings_arr[assignments == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    # Build cluster lists
    clusters: list[list[int]] = [[] for _ in range(k)]
    for i, cluster_id in enumerate(assignments):
        clusters[cluster_id].append(i)
    
    return [c for c in clusters if c]  # Remove empty clusters


def trim_to_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


_CODE_SIGNATURE_RE = re.compile(
    r"^(?:"
    r"(?:export\s+)?(?:async\s+)?(?:def|function|fn|func|fun|sub)\s+\w+[^{;]*"
    r"|(?:public|private|protected|internal|static|abstract|final|override|open|sealed|suspend|inline)\s+.*\w+\s*\([^)]*\)"
    r"|class\s+\w+[^{]*"
    r"|interface\s+\w+[^{]*"
    r"|struct\s+\w+[^{]*"
    r"|enum\s+\w+[^{]*"
    r"|trait\s+\w+[^{]*"
    r"|impl\s+.*"
    r"|type\s+\w+\s+(?:struct|interface)[^{]*"
    r"|module\s+\w+"
    r"|(?:const|let|var)\s+\w+\s*[:=]"
    r")",
    re.MULTILINE,
)

_DOCSTRING_RE = re.compile(
    r'(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|/\*\*[\s\S]*?\*/|///.*|//!.*|##.*)',
)


def extract_code_signatures(text: str) -> list[str]:
    """Extract function/class signatures and docstrings from code text."""
    results: list[str] = []
    for match in _CODE_SIGNATURE_RE.finditer(text):
        sig = match.group(0).strip()
        if sig and len(sig.split()) >= 2:
            results.append(sig)
    for match in _DOCSTRING_RE.finditer(text):
        doc = match.group(0).strip()
        # Trim long docstrings to first 3 lines
        doc_lines = doc.splitlines()
        if len(doc_lines) > 3:
            doc = "\n".join(doc_lines[:3]) + " ..."
        if doc and len(doc.split()) >= 3:
            results.append(doc)
    return results


def merge_adjacent_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Merge candidates with adjacent chunk indices from the same source to eliminate overlap."""
    if len(candidates) <= 1:
        return candidates

    # Group by source
    by_source: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_source.setdefault(c.source, []).append(c)

    merged: list[Candidate] = []
    for source, group in by_source.items():
        group.sort(key=lambda c: c.chunk_index)
        run: list[Candidate] = [group[0]]

        for i in range(1, len(group)):
            prev = run[-1]
            curr = group[i]
            if curr.chunk_index == prev.chunk_index + 1:
                run.append(curr)
            else:
                merged.append(_merge_run(run))
                run = [curr]
        merged.append(_merge_run(run))

    # Preserve original score ordering
    merged.sort(key=lambda c: c.final_score, reverse=True)
    return merged


def _merge_run(run: list[Candidate]) -> Candidate:
    if len(run) == 1:
        return run[0]
    combined_text = "\n".join(c.text for c in run)
    return Candidate(
        chunk_id=run[0].chunk_id,
        source=run[0].source,
        chunk_index=run[0].chunk_index,
        text=combined_text,
        token_estimate=sum(c.token_estimate for c in run),
        bm25_score=run[0].bm25_score,
        fts_rank=run[0].fts_rank,
        vector_rank=run[0].vector_rank,
        vector_score=max(c.vector_score for c in run),
        fts_score=max(c.fts_score for c in run),
        overlap_score=max(c.overlap_score for c in run),
        final_score=max(c.final_score for c in run),
    )


def compress_candidates(query: str, candidates: list[Candidate], word_budget: int) -> list[str]:
    # Merge adjacent chunks before compression to eliminate overlap redundancy
    candidates = merge_adjacent_candidates(candidates)

    query_terms = set(tokenize(query))
    bullets: list[str] = []
    words_used = 0

    # Get configurable weights for sentence scoring
    length_bonus_w = get_weight("sentence_length_bonus_weight")
    length_normalizer = get_weight("sentence_length_normalizer")

    for candidate in candidates:
        source_is_code = is_code_file(candidate.source)

        if source_is_code:
            # For code: extract signatures and docstrings instead of sentence splitting
            snippets = extract_code_signatures(candidate.text)
            if not snippets:
                # Fallback: use first N lines preserving code structure
                code_lines = [l for l in candidate.text.splitlines() if l.strip()]
                snippets = code_lines[:5] if code_lines else []
            if not snippets:
                continue
            summary = " | ".join(snippets[:4]).strip()
        else:
            # For prose: TextRank-based sentence scoring for intelligent summarization
            sentences = split_sentences(candidate.text)
            if not sentences:
                continue
            
            # Get configurable weights
            textrank_w = get_weight("textrank_weight")
            query_relevance_w = get_weight("query_relevance_weight")
            
            # Use TextRank for semantic importance, boosted by query overlap
            textrank_scores = textrank_score_sentences(sentences)
            scored_sentences: list[tuple[float, str]] = []
            
            for tr_score, sentence in textrank_scores:
                sent_terms = set(tokenize(sentence))
                overlap = len(query_terms & sent_terms)
                query_signal = overlap / float(len(query_terms)) if query_terms else 0.0
                length_bonus = min(1.0, len(sentence.split()) / length_normalizer)
                # Combine TextRank centrality with query relevance
                combined_score = (textrank_w * tr_score) + (query_relevance_w * query_signal) + (length_bonus_w * length_bonus)
                scored_sentences.append((combined_score, sentence))
            
            scored_sentences.sort(key=lambda pair: pair[0], reverse=True)
            selected = [s for _, s in scored_sentences[:2]]
            summary = " ".join(selected).strip()

        if not summary:
            continue

        citation = f"[{Path(candidate.source).name}#chunk-{candidate.chunk_index}]"
        bullet = f"{summary} {citation}".strip()
        bullet_words = len(bullet.split())

        if bullets and words_used + bullet_words > word_budget:
            continue
        if not bullets and bullet_words > word_budget:
            bullet = trim_to_words(bullet, max_words=word_budget)
            bullet_words = len(bullet.split())

        bullets.append(bullet)
        words_used += bullet_words
        if words_used >= word_budget:
            break

    if bullets:
        return bullets

    fallback: list[str] = []
    for candidate in candidates[:3]:
        snippet = " ".join(candidate.text.split()[:35]).strip()
        if snippet:
            fallback.append(f"{snippet}... [{Path(candidate.source).name}#chunk-{candidate.chunk_index}]")
    return fallback


def build_packet(
    query: str,
    selected: list[Candidate],
    candidate_pool: list[Candidate],
    bullets: list[str],
    fts_hit_count: int,
    vector_hit_count: int,
    hybrid_mode: str,
    retrieval_mode: str,
    vector_backend_used: str,
    vector_model_used: str | None,
    vector_retrieval_path: str,
    referenced_symbols: list[dict] | None = None,
    imported_context: list[Candidate] | None = None,
) -> dict:
    source_count = len({c.source for c in selected})
    candidate_pool_tokens = sum(c.token_estimate for c in candidate_pool)
    selected_token_estimate = sum(c.token_estimate for c in selected)
    compressed_token_estimate = sum(estimate_tokens(b) for b in bullets)
    estimated_savings = max(0, selected_token_estimate - compressed_token_estimate)
    savings_from_pool = max(0, candidate_pool_tokens - compressed_token_estimate)
    savings_pct_pool = (
        round((savings_from_pool / candidate_pool_tokens) * 100.0, 2)
        if candidate_pool_tokens > 0
        else 0.0
    )

    lines: list[str] = [
        "CONTEXT_PACKET_START",
        f"query: {query}",
        f"selected_chunks: {len(selected)}",
        f"sources: {source_count}",
        f"retrieval_mode: {retrieval_mode}",
        f"hybrid_mode: {hybrid_mode}",
        "fts_ranking: bm25",
        f"vector_backend_used: {vector_backend_used}",
        f"vector_retrieval_path: {vector_retrieval_path}",
    ]
    if vector_model_used:
        lines.append(f"vector_model_used: {vector_model_used}")
    lines.extend(
        [
            f"fts_hits: {fts_hit_count}",
            f"vector_hits: {vector_hit_count}",
            f"estimated_token_savings: {estimated_savings}",
            f"candidate_pool_token_reduction_pct: {savings_pct_pool}",
            "",
            "compressed_context:",
        ]
    )
    for bullet in bullets:
        lines.append(f"- {bullet}")
    lines.extend(["", "CONTEXT_PACKET_END"])

    return {
        "query": query,
        "selected_chunks": len(selected),
        "source_count": source_count,
        "estimated_token_savings": estimated_savings,
        "retrieval": {
            "mode": retrieval_mode,
            "hybrid_mode": hybrid_mode,
            "bm25_enabled": True,
            "fts_hits": fts_hit_count,
            "vector_hits": vector_hit_count,
            "vector_backend_used": vector_backend_used,
            "vector_model_used": vector_model_used,
            "vector_retrieval_path": vector_retrieval_path,
            "candidate_pool_size": len(candidate_pool),
        },
        "token_metrics": {
            "candidate_pool_tokens": candidate_pool_tokens,
            "selected_chunk_tokens": selected_token_estimate,
            "compressed_tokens": compressed_token_estimate,
            "payload_tokens": compressed_token_estimate,
            "savings_from_selected_tokens": estimated_savings,
            "savings_from_candidate_pool_tokens": savings_from_pool,
            "savings_from_candidate_pool_pct": savings_pct_pool,
        },
        "bullets": bullets,
        "candidates": [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "bm25_score": round(float(c.bm25_score), 6) if c.bm25_score is not None else None,
                "fts_score": round(c.fts_score, 6),
                "vector_score": round(c.vector_score, 6),
                "overlap_score": round(c.overlap_score, 6),
                "final_score": round(c.final_score, 6),
                "fts_rank": c.fts_rank,
                "vector_rank": c.vector_rank,
            }
            for c in selected
        ],
        "packet": "\n".join(lines),
    }


def run_retrieval_pipeline(
    conn: sqlite3.Connection,
    db_path: Path,
    query: str,
    top_k: int,
    fts_k: int,
    vector_k: int,
    min_fts_hits: int,
    hybrid_mode: str,
    retrieval_mode: str,
    embedding_backend: str,
    embedding_model: str | None,
    session_id: str,
    query_cache_ttl_seconds: int,
    dimensions: int,
    word_budget: int,
) -> dict:
    now_epoch = utc_now_epoch()
    cleanup_query_cache(conn=conn, now_epoch=now_epoch)

    index_fingerprint = get_index_fingerprint(conn)
    cache_key_material = json.dumps(
        {
            "query": query,
            "top_k": top_k,
            "fts_k": fts_k,
            "vector_k": vector_k,
            "min_fts_hits": min_fts_hits,
            "hybrid_mode": hybrid_mode,
            "retrieval_mode": retrieval_mode,
            "embedding_backend": embedding_backend,
            "embedding_model": embedding_model,
            "dimensions": dimensions,
            "word_budget": word_budget,
            "session_id": session_id,
            "index_fingerprint": index_fingerprint,
        },
        sort_keys=True,
    )
    query_cache_key = hash_text(cache_key_material)
    cached_result = get_cached_query_result(conn=conn, cache_key=query_cache_key, now_epoch=now_epoch)
    if cached_result is not None:
        memory_path = session_memory_path(db_path)
        updated_memory = update_session_memory(
            memory_path=memory_path,
            session_id=session_id,
            query=query,
            selected_sources=[str(item.get("source")) for item in cached_result.get("candidates", []) if isinstance(item, dict)],
        )
        cached_result["session_memory"] = {
            "session_id": session_id,
            "recent_queries": get_recent_session_queries(updated_memory, session_id=session_id, limit=4),
        }
        cached_result.setdefault("cache", {})
        cached_result["cache"]["hit"] = True
        cached_result["cache"]["key"] = query_cache_key
        cached_result["cache"].setdefault("ttl_seconds", query_cache_ttl_seconds)
        return cached_result

    fts_hits = fts_retrieve(conn, query, limit=fts_k)
    vector_hits: list[Candidate] = []
    vector_backend_used = "disabled"
    vector_model_used: str | None = None
    vector_retrieval_path = "disabled"

    # Skip vector retrieval when using hash embeddings (redundant with FTS5/BM25)
    use_vector = hybrid_mode == "always" or (hybrid_mode == "fallback" and len(fts_hits) < min_fts_hits)
    if embedding_backend == "hash" and should_skip_vector_for_hash():
        use_vector = False
        vector_retrieval_path = "skipped_hash_backend"

    if use_vector:
        vector_hits, vector_backend_used, vector_model_used, vector_retrieval_path = vector_retrieve(
            conn=conn,
            db_path=db_path,
            query=query,
            limit=vector_k,
            dimensions=dimensions,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
        )

    selected, candidate_pool = rerank_candidates(
        query=query,
        fts_hits=fts_hits,
        vector_hits=vector_hits,
        top_k=top_k,
    )
    bullets = compress_candidates(query=query, candidates=selected, word_budget=word_budget)

    packet = build_packet(
        query=query,
        selected=selected,
        candidate_pool=candidate_pool,
        bullets=bullets,
        fts_hit_count=len(fts_hits),
        vector_hit_count=len(vector_hits),
        hybrid_mode=hybrid_mode,
        retrieval_mode=retrieval_mode,
        vector_backend_used=vector_backend_used,
        vector_model_used=vector_model_used,
        vector_retrieval_path=vector_retrieval_path,
    )

    memory_path = session_memory_path(db_path)
    updated_memory = update_session_memory(
        memory_path=memory_path,
        session_id=session_id,
        query=query,
        selected_sources=[c.source for c in selected],
    )
    packet["session_memory"] = {
        "session_id": session_id,
        "recent_queries": get_recent_session_queries(updated_memory, session_id=session_id, limit=4),
    }
    packet.setdefault("cache", {})
    packet["cache"]["hit"] = False
    packet["cache"]["key"] = query_cache_key
    packet["cache"]["ttl_seconds"] = query_cache_ttl_seconds

    set_cached_query_result(
        conn=conn,
        cache_key=query_cache_key,
        payload=packet,
        now_epoch=now_epoch,
        ttl_seconds=query_cache_ttl_seconds,
    )

    return packet


def cmd_index(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser().resolve()
    conn = connect_db(db_path)
    try:
        files = collect_input_files(args.inputs)
        if not files:
            print("No indexable files found.", file=sys.stderr)
            return 1

        embedding_backend, embedding_model = resolve_embedding_backend(
            requested_backend=args.embedding_backend,
            requested_model=args.embedding_model,
        )
        chunk_size_words, overlap_words = normalize_chunking(args.chunk_size, args.overlap)

        stats = index_corpus(
            conn=conn,
            file_paths=files,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            dimensions=args.dimensions,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
        )

        inferred_dims = infer_embedding_dimensions(
            conn=conn,
            backend=embedding_backend,
            model_name=embedding_model,
            fallback_dimensions=args.dimensions,
        )
        ann_built = build_hnsw_index(
            conn=conn,
            db_path=db_path,
            backend=embedding_backend,
            dimensions=inferred_dims,
            model_name=embedding_model,
        )

        print(
            json.dumps(
                {
                    "db": str(db_path),
                    "files_indexed": stats["files_indexed"],
                    "chunks_indexed": stats["chunks_indexed"],
                    "embedding_backend": embedding_backend,
                    "embedding_model": embedding_model,
                    "ann_engine": DEFAULT_ANN_ENGINE,
                    "ann_index_built": ann_built,
                },
                indent=2,
            )
        )
        return 0
    finally:
        conn.close()


def cmd_query(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser().resolve()
    conn = connect_db(db_path)
    try:
        query_ok, query_error = validate_query_input(args.query)
        if not query_ok:
            print(query_error, file=sys.stderr)
            return 2

        embedding_backend, embedding_model = resolve_embedding_backend(
            requested_backend=args.embedding_backend,
            requested_model=args.embedding_model,
        )

        result = run_retrieval_pipeline(
            conn=conn,
            db_path=db_path,
            query=args.query,
            top_k=args.top_k,
            fts_k=args.fts_k,
            vector_k=args.vector_k,
            min_fts_hits=args.min_fts_hits,
            hybrid_mode=args.hybrid_mode,
            retrieval_mode=args.mode,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            session_id=args.session_id,
            query_cache_ttl_seconds=args.query_cache_ttl,
            dimensions=args.dimensions,
            word_budget=args.word_budget,
        )

        print(json.dumps(result, indent=2) if args.json else result["packet"])
        return 0
    finally:
        conn.close()


def cmd_run(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser().resolve()
    conn = connect_db(db_path)
    try:
        query_ok, query_error = validate_query_input(args.query)
        if not query_ok:
            print(query_error, file=sys.stderr)
            return 2

        embedding_backend, embedding_model = resolve_embedding_backend(
            requested_backend=args.embedding_backend,
            requested_model=args.embedding_model,
        )

        chunk_size_words, overlap_words = normalize_chunking(args.chunk_size, args.overlap)

        if args.inputs:
            files = collect_input_files(args.inputs)
            if not files:
                print("No indexable files found for run mode.", file=sys.stderr)
                return 1
            index_corpus(
                conn=conn,
                file_paths=files,
                chunk_size_words=chunk_size_words,
                overlap_words=overlap_words,
                dimensions=args.dimensions,
                embedding_backend=embedding_backend,
                embedding_model=embedding_model,
            )
            inferred_dims = infer_embedding_dimensions(
                conn=conn,
                backend=embedding_backend,
                model_name=embedding_model,
                fallback_dimensions=args.dimensions,
            )
            build_hnsw_index(
                conn=conn,
                db_path=db_path,
                backend=embedding_backend,
                dimensions=inferred_dims,
                model_name=embedding_model,
            )

        result = run_retrieval_pipeline(
            conn=conn,
            db_path=db_path,
            query=args.query,
            top_k=args.top_k,
            fts_k=args.fts_k,
            vector_k=args.vector_k,
            min_fts_hits=args.min_fts_hits,
            hybrid_mode=args.hybrid_mode,
            retrieval_mode=args.mode,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            session_id=args.session_id,
            query_cache_ttl_seconds=args.query_cache_ttl,
            dimensions=args.dimensions,
            word_budget=args.word_budget,
        )

        print(json.dumps(result, indent=2) if args.json else result["packet"])
        return 0
    finally:
        conn.close()


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run comprehensive benchmarks and generate metrics report."""
    import time
    
    db_path = Path(args.db).expanduser().resolve()
    
    # Collect test files
    files = collect_input_files(args.inputs) if args.inputs else []
    if not files:
        print("No input files for benchmarking. Provide --inputs.", file=sys.stderr)
        return 1
    
    # Calculate raw input stats
    raw_tokens_total = 0
    raw_chars_total = 0
    file_stats: list[dict] = []
    
    for file_path in files:
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="replace")
            tokens = estimate_tokens(content)
            raw_tokens_total += tokens
            raw_chars_total += len(content)
            file_stats.append({
                "file": str(file_path),
                "tokens": tokens,
                "chars": len(content),
            })
        except Exception:
            continue
    
    # Benchmark indexing
    conn = connect_db(db_path)
    embedding_backend, embedding_model = resolve_embedding_backend(
        requested_backend=args.embedding_backend,
        requested_model=args.embedding_model,
    )
    chunk_size_words, overlap_words = normalize_chunking(args.chunk_size, args.overlap)
    
    index_start = time.perf_counter()
    stats = index_corpus(
        conn=conn,
        file_paths=files,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
        dimensions=args.dimensions,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
    )
    index_latency_ms = (time.perf_counter() - index_start) * 1000
    
    # Build ANN index
    inferred_dims = infer_embedding_dimensions(
        conn=conn,
        backend=embedding_backend,
        model_name=embedding_model,
        fallback_dimensions=args.dimensions,
    )
    build_hnsw_index(
        conn=conn,
        db_path=db_path,
        backend=embedding_backend,
        dimensions=inferred_dims,
        model_name=embedding_model,
    )
    
    # Benchmark queries
    test_queries = [
        "How does authentication work?",
        "What functions handle database connections?",
        "Error handling patterns in this codebase",
        "Main entry point and initialization",
        "API endpoint implementations",
    ]
    
    query_metrics: list[dict] = []
    compressed_tokens_total = 0
    
    for query in test_queries:
        query_start = time.perf_counter()
        result = run_retrieval_pipeline(
            conn=conn,
            db_path=db_path,
            query=query,
            top_k=args.top_k,
            fts_k=args.fts_k,
            vector_k=args.vector_k,
            min_fts_hits=DEFAULT_MIN_FTS_HITS,
            hybrid_mode=args.hybrid_mode,
            retrieval_mode="compact",
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            session_id="benchmark",
            query_cache_ttl_seconds=0,  # Disable cache for benchmarking
            dimensions=args.dimensions,
            word_budget=args.word_budget,
        )
        query_latency_ms = (time.perf_counter() - query_start) * 1000
        
        compressed_tokens = result.get("token_metrics", {}).get("compressed_tokens", 0)
        selected_tokens = result.get("token_metrics", {}).get("selected_chunk_tokens", 0)
        compressed_tokens_total += compressed_tokens
        
        query_metrics.append({
            "query": query,
            "latency_ms": round(query_latency_ms, 2),
            "fts_hits": result.get("retrieval", {}).get("fts_hits", 0),
            "vector_hits": result.get("retrieval", {}).get("vector_hits", 0),
            "selected_chunks": result.get("selected_chunks", 0),
            "selected_tokens": selected_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": round(selected_tokens / max(1, compressed_tokens), 2),
        })
    
    conn.close()
    
    # Calculate aggregate metrics
    avg_query_latency = sum(q["latency_ms"] for q in query_metrics) / len(query_metrics)
    avg_compression_ratio = sum(q["compression_ratio"] for q in query_metrics) / len(query_metrics)
    
    # Token savings analysis
    # If user sent all raw content vs using token-reducer
    hypothetical_saved = raw_tokens_total - (compressed_tokens_total / len(test_queries))
    savings_pct = (hypothetical_saved / raw_tokens_total * 100) if raw_tokens_total > 0 else 0
    
    # Cost savings estimate (assuming $0.01 per 1K input tokens for Claude)
    cost_per_1k_tokens = 0.01
    raw_cost = (raw_tokens_total / 1000) * cost_per_1k_tokens
    compressed_cost = (compressed_tokens_total / len(test_queries) / 1000) * cost_per_1k_tokens
    cost_savings = raw_cost - compressed_cost
    
    report = {
        "benchmark_summary": {
            "files_indexed": len(files),
            "chunks_created": stats["chunks_indexed"],
            "raw_input_tokens": raw_tokens_total,
            "raw_input_chars": raw_chars_total,
            "embedding_backend": embedding_backend,
            "embedding_model": embedding_model,
        },
        "latency_metrics": {
            "index_time_ms": round(index_latency_ms, 2),
            "avg_query_time_ms": round(avg_query_latency, 2),
            "min_query_time_ms": round(min(q["latency_ms"] for q in query_metrics), 2),
            "max_query_time_ms": round(max(q["latency_ms"] for q in query_metrics), 2),
        },
        "compression_metrics": {
            "avg_compression_ratio": round(avg_compression_ratio, 2),
            "avg_compressed_tokens_per_query": round(compressed_tokens_total / len(test_queries), 0),
            "total_queries_benchmarked": len(test_queries),
        },
        "token_savings": {
            "raw_context_tokens": raw_tokens_total,
            "avg_compressed_tokens": round(compressed_tokens_total / len(test_queries), 0),
            "estimated_savings_pct": round(savings_pct, 1),
            "tokens_saved_per_query": round(hypothetical_saved, 0),
        },
        "cost_analysis": {
            "note": "Estimated at $0.01 per 1K input tokens",
            "raw_context_cost_usd": round(raw_cost, 4),
            "compressed_cost_usd": round(compressed_cost, 4),
            "savings_per_query_usd": round(cost_savings, 4),
            "savings_per_100_queries_usd": round(cost_savings * 100, 2),
        },
        "query_details": query_metrics,
    }
    
    print(json.dumps(report, indent=2))
    
    # Print human-readable summary
    print("\n" + "=" * 60, file=sys.stderr)
    print("TOKEN REDUCER BENCHMARK SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Files indexed: {len(files)}", file=sys.stderr)
    print(f"Raw input tokens: {raw_tokens_total:,}", file=sys.stderr)
    print(f"Avg compressed tokens: {compressed_tokens_total // len(test_queries):,}", file=sys.stderr)
    print(f"Token savings: {savings_pct:.1f}%", file=sys.stderr)
    print(f"Avg query latency: {avg_query_latency:.1f}ms", file=sys.stderr)
    print(f"Cost savings per 100 queries: ${cost_savings * 100:.2f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    return 0


def cmd_self_test(_args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="token-reducer-self-test-") as tmp_dir:
        root = Path(tmp_dir)
        db_path = root / "index.db"

        sample_files = {
            "auth_middleware.md": """
            The authentication middleware checks JWT tokens in cookies.
            If the token is invalid, the middleware redirects to /login.
            It runs before protected route handlers.
            """,
            "session_guard.md": """
            Authorization guard validates session credentials prior to route execution.
            The guard denies unauthorized access and records audit logs.
            """,
            "unrelated_notes.md": """
            CSS palette and typography notes for landing page.
            Icon sizing and spacing guidance for components.
            """,
        }

        for name, content in sample_files.items():
            (root / name).write_text(content.strip() + "\n", encoding="utf-8")

        conn = connect_db(db_path)
        try:
            files = collect_input_files([str(root)])
            index_corpus(
                conn=conn,
                file_paths=files,
                chunk_size_words=DEFAULT_CHUNK_SIZE,
                overlap_words=DEFAULT_CHUNK_OVERLAP,
                dimensions=DEFAULT_DIMENSIONS,
                embedding_backend="hash",
                embedding_model=None,
            )
            result = run_retrieval_pipeline(
                conn=conn,
                db_path=db_path,
                query="How does middleware validate auth tokens before protected routes?",
                top_k=5,
                fts_k=8,
                vector_k=8,
                min_fts_hits=DEFAULT_MIN_FTS_HITS,
                hybrid_mode="always",
                retrieval_mode="compact",
                embedding_backend="hash",
                embedding_model=None,
                session_id="self-test",
                query_cache_ttl_seconds=DEFAULT_QUERY_CACHE_TTL_SECONDS,
                dimensions=DEFAULT_DIMENSIONS,
                word_budget=DEFAULT_WORD_BUDGET,
            )

            result_repeat = run_retrieval_pipeline(
                conn=conn,
                db_path=db_path,
                query="How does middleware validate auth tokens before protected routes?",
                top_k=5,
                fts_k=8,
                vector_k=8,
                min_fts_hits=DEFAULT_MIN_FTS_HITS,
                hybrid_mode="always",
                retrieval_mode="compact",
                embedding_backend="hash",
                embedding_model=None,
                session_id="self-test",
                query_cache_ttl_seconds=DEFAULT_QUERY_CACHE_TTL_SECONDS,
                dimensions=DEFAULT_DIMENSIONS,
                word_budget=DEFAULT_WORD_BUDGET,
            )
        finally:
            conn.close()

    checks = {
        "bm25_enabled": bool(result["retrieval"]["bm25_enabled"]),
        "fts_hits_gt_zero": int(result["retrieval"]["fts_hits"]) > 0,
        "vector_hits_gt_zero": int(result["retrieval"]["vector_hits"]) > 0,
        "selected_chunks_le_5": int(result["selected_chunks"]) <= 5,
        "compressed_le_selected_tokens": int(result["token_metrics"]["compressed_tokens"])
        <= int(result["token_metrics"]["selected_chunk_tokens"]),
        "cache_hit_on_repeat": bool(result_repeat.get("cache", {}).get("hit", False)),
        "session_memory_present": bool(result_repeat.get("session_memory", {}).get("recent_queries")),
    }

    ok = all(checks.values())
    summary = {
        "ok": ok,
        "checks": checks,
        "retrieval": result["retrieval"],
        "token_metrics": result["token_metrics"],
        "repeat_query_cache": result_repeat.get("cache", {}),
    }
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="context_pipeline",
        description="Local-first context slimming pipeline for Claude Code.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Preprocess and index input files.")
    index_parser.add_argument("--inputs", nargs="+", required=True, help="Files or directories to index.")
    index_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite index path.")
    index_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    index_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    index_parser.add_argument("--embedding-backend", choices=["hash", "ml"], default=DEFAULT_EMBEDDING_BACKEND)
    index_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    index_parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)

    query_parser = subparsers.add_parser("query", help="Retrieve and compress from existing index.")
    query_parser.add_argument("--query", required=True, help="Question or objective.")
    query_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite index path.")
    query_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    query_parser.add_argument("--fts-k", type=int, default=DEFAULT_FTS_K)
    query_parser.add_argument("--vector-k", type=int, default=DEFAULT_VECTOR_K)
    query_parser.add_argument("--min-fts-hits", type=int, default=DEFAULT_MIN_FTS_HITS)
    query_parser.add_argument("--mode", choices=["compact", "deep-debug"], default=DEFAULT_RETRIEVAL_MODE)
    query_parser.add_argument("--hybrid-mode", choices=["always", "fallback"], default=DEFAULT_HYBRID_MODE)
    query_parser.add_argument("--embedding-backend", choices=["hash", "ml"], default=DEFAULT_EMBEDDING_BACKEND)
    query_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    query_parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    query_parser.add_argument("--session-id", default="default")
    query_parser.add_argument("--query-cache-ttl", type=int, default=DEFAULT_QUERY_CACHE_TTL_SECONDS)
    query_parser.add_argument("--word-budget", type=int, default=DEFAULT_WORD_BUDGET)
    query_parser.add_argument("--json", action="store_true")

    run_parser = subparsers.add_parser("run", help="End-to-end run: optional indexing, then retrieval and compression.")
    run_parser.add_argument("--inputs", nargs="*", default=[], help="Files or directories to index first.")
    run_parser.add_argument("--query", required=True, help="Question or objective.")
    run_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite index path.")
    run_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    run_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    run_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    run_parser.add_argument("--fts-k", type=int, default=DEFAULT_FTS_K)
    run_parser.add_argument("--vector-k", type=int, default=DEFAULT_VECTOR_K)
    run_parser.add_argument("--min-fts-hits", type=int, default=DEFAULT_MIN_FTS_HITS)
    run_parser.add_argument("--mode", choices=["compact", "deep-debug"], default=DEFAULT_RETRIEVAL_MODE)
    run_parser.add_argument("--hybrid-mode", choices=["always", "fallback"], default=DEFAULT_HYBRID_MODE)
    run_parser.add_argument("--embedding-backend", choices=["hash", "ml"], default=DEFAULT_EMBEDDING_BACKEND)
    run_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    run_parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    run_parser.add_argument("--session-id", default="default")
    run_parser.add_argument("--query-cache-ttl", type=int, default=DEFAULT_QUERY_CACHE_TTL_SECONDS)
    run_parser.add_argument("--word-budget", type=int, default=DEFAULT_WORD_BUDGET)
    run_parser.add_argument("--json", action="store_true")

    subparsers.add_parser("self-test", help="Run internal checks for BM25+hybrid retrieval and token reduction behavior.")

    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks and generate metrics report.")
    benchmark_parser.add_argument("--inputs", nargs="+", required=True, help="Files or directories to benchmark.")
    benchmark_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite index path.")
    benchmark_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    benchmark_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    benchmark_parser.add_argument("--embedding-backend", choices=["hash", "ml"], default=DEFAULT_EMBEDDING_BACKEND)
    benchmark_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    benchmark_parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    benchmark_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    benchmark_parser.add_argument("--fts-k", type=int, default=DEFAULT_FTS_K)
    benchmark_parser.add_argument("--vector-k", type=int, default=DEFAULT_VECTOR_K)
    benchmark_parser.add_argument("--hybrid-mode", choices=["always", "fallback"], default=DEFAULT_HYBRID_MODE)
    benchmark_parser.add_argument("--word-budget", type=int, default=DEFAULT_WORD_BUDGET)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        return cmd_index(args)
    if args.command == "query":
        return cmd_query(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "self-test":
        return cmd_self_test(args)
    if args.command == "benchmark":
        return cmd_benchmark(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
