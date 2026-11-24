"""Layer-3 token intelligence: dedupe + light structural cleanup (not blind truncation)."""

from __future__ import annotations

import re

from .chunker import estimate_tokens

try:
    import token_reducer_core as _rc  # type: ignore

    _collapse_blank_lines = _rc.collapse_blank_lines
except (ImportError, AttributeError):  # older extension builds without new symbols

    def _collapse_blank_lines(text: str) -> str:
        lines = text.splitlines()
        out: list[str] = []
        prev_blank = False
        for line in lines:
            blank = not line.strip()
            if blank and prev_blank:
                continue
            prev_blank = blank
            out.append(line)
        return "\n".join(out)


_BULLET_PREFIX = re.compile(r"^\s*[-*]\s*")


def _normalize_for_dedupe(line: str) -> str:
    s = line.strip().lower()
    s = _BULLET_PREFIX.sub("", s)
    return " ".join(s.split())[:480]


def refine_output_bullets(bullets: list[str], query: str) -> list[str]:
    """Remove near-duplicate bullets and collapse redundant vertical space in long items."""
    _ = query  # reserved for query-weighted pruning later
    seen: set[str] = set()
    out: list[str] = []
    for b in bullets:
        raw = b.strip()
        if not raw:
            continue
        key = _normalize_for_dedupe(raw)
        if key in seen:
            continue
        seen.add(key)
        collapsed = _collapse_blank_lines(raw)
        if len(collapsed) > 2400 and estimate_tokens(collapsed) > 320:
            first = "\n".join(collapsed.splitlines()[:24])
            out.append(first + "\n…")
        else:
            out.append(collapsed)
    return out
