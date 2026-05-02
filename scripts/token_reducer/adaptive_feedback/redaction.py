"""Privacy helpers: hashes and bounded diagnostic strings (design spec Section 2.3)."""

from __future__ import annotations

import hashlib


def hash_text(text: str, *, nibble_len: int = 16) -> str:
    """Stable SHA-256 hex digest truncated for correlation ids (not cryptographic secrecy)."""
    h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return h[:nibble_len]


def bound_diagnostic(message: str, *, max_len: int = 120) -> str:
    """Truncate single-line diagnostic for event payloads."""
    one = message.replace("\n", " ").replace("\r", " ").strip()
    if len(one) <= max_len:
        return one
    return one[: max_len - 1] + "…"
