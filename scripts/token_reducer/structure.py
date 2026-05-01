"""Structure-aware chunk metadata (local inference, no network)."""

from __future__ import annotations

import json
import re
from pathlib import Path


def infer_chunk_meta(chunk: str, source: str) -> dict[str, str | int]:
    """Infer symbol name, chunk type, and approximate line span from chunk text."""
    ext = Path(source).suffix.lower()
    lines = chunk.splitlines()
    n = max(1, len(lines))
    head = "\n".join(lines[:8])
    symbol = ""
    kind: str = "text"

    if ext == ".py":
        m = re.search(r"^(?:async\s+)?def\s+(\w+)\s*\(", head, re.M)
        if m:
            symbol, kind = m.group(1), "function"
        else:
            m = re.search(r"^class\s+(\w+)\b", head, re.M)
            if m:
                symbol, kind = m.group(1), "class"
    elif ext in {".ts", ".tsx", ".js", ".jsx"}:
        m = re.search(r"(?:async\s+)?function\s+(\w+)\s*\(", head, re.M)
        if m:
            symbol, kind = m.group(1), "function"
        else:
            m = re.search(r"^class\s+(\w+)\b", head, re.M)
            if m:
                symbol, kind = m.group(1), "class"
    elif ext == ".go":
        m = re.search(r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(", head, re.M)
        if m:
            symbol, kind = m.group(1), "function"
        else:
            m = re.search(r"type\s+(\w+)\s+struct", head, re.M)
            if m:
                symbol, kind = m.group(1), "class"
    elif ext == ".rs":
        m = re.search(r"(?:pub\s+)?fn\s+(\w+)\s*\(", head, re.M)
        if m:
            symbol, kind = m.group(1), "function"
        else:
            m = re.search(r"(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)", head, re.M)
            if m:
                symbol, kind = m.group(1), "class"

    if not symbol:
        symbol = Path(source).stem

    return {
        "file_path": source,
        "symbol_name": symbol,
        "chunk_type": kind,
        "start_line": 1,
        "end_line": n,
    }


def meta_json_from_chunk(chunk: str, source: str) -> str:
    return json.dumps(infer_chunk_meta(chunk, source), separators=(",", ":"))
