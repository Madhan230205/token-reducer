"""Canonical stability hashes (spec Section 6.2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

BulletOrder = Literal["as_is", "sorted"]


def _norm_source_path(source: str, workspace_root: Path | None) -> str:
    p = Path(source).resolve()
    if workspace_root is not None:
        try:
            rel = p.relative_to(workspace_root.resolve())
            return rel.as_posix().replace("\\", "/").lstrip("./")
        except ValueError:
            pass
    name = p.name
    # Windows drive normalization hint for cross-platform compares
    return (
        name.lower() if len(name) == len(source) else p.as_posix().replace("\\", "/").lstrip("./")
    )


def canonical_selected_sources_hash(
    sources: list[str],
    *,
    workspace_root: Path | None,
) -> str:
    normed = sorted({_norm_source_path(s, workspace_root) for s in sources})
    body = "\n".join(normed)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def canonical_bullets_hash(
    bullets: list[str],
    *,
    bullet_order: BulletOrder,
    collapse_ws: bool,
) -> str:
    lines = list(bullets)
    if bullet_order == "sorted":
        lines = sorted(lines)
    out_lines: list[str] = []
    for line in lines:
        s = line.replace("\r\n", "\n").replace("\r", "\n")
        s = s.rstrip()
        if collapse_ws:
            s = " ".join(s.split())
        out_lines.append(s)
    body = "\n".join(out_lines) + "\n"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def plugin_subset_digest(payload: dict[str, Any], keys: list[str]) -> str:
    subset = {k: payload[k] for k in sorted(keys) if k in payload}
    raw = json.dumps(subset, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
