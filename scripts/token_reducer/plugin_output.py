"""Claude plugin JSON payload (structured context)."""

from __future__ import annotations

import json

from .compressor import denoise_code_for_plugin
from .intent import IntentType
from .models import Candidate
from .structure import infer_chunk_meta
from .summarizer import summarize_context, why_relevant


def build_claude_plugin_payload(
    query: str,
    intent: IntentType,
    candidates: list[Candidate],
    *,
    patch_first: bool = False,
    context_strategy: dict[str, object] | None = None,
) -> dict[str, object]:
    summary, relationships = summarize_context(query, intent, candidates)
    files = sorted({c.source for c in candidates})
    code_context: list[dict[str, object]] = []
    # Dedupe by chunk identity only. (file, symbol_name) was wrong: infer_chunk_meta
    # falls back to Path(source).stem for non-symbol-leading chunks, so multiple
    # prose/config segments from one file collapsed to a single entry.
    seen_chunk_ids: set[int] = set()
    for c in candidates:
        if c.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(c.chunk_id)
        meta = infer_chunk_meta(c.text, c.source)
        code = denoise_code_for_plugin(c.text, query)
        max_code = 5200 if patch_first else 12000
        if len(code) > max_code:
            code = code[:max_code] + "\n…"
        code_context.append(
            {
                "file": c.source,
                "symbol": str(meta.get("symbol_name", "")),
                "chunk_type": str(meta.get("chunk_type", "text")),
                "start_line": int(meta.get("start_line", 0)),
                "end_line": int(meta.get("end_line", 0)),
                "code": code,
                "why_relevant": why_relevant(c, query, intent),
            }
        )
    if context_strategy:
        af = context_strategy.get("attention_frame")
        if isinstance(af, str) and (t := af.strip()):
            # Fold framing into the narrative summary — no extra top-level keys.
            summary = f"{t}\n\n{summary}" if summary else t
    payload: dict[str, object] = {
        "intent": intent,
        "summary": summary,
        "relevant_files": files,
        "relationships": relationships,
        "code_context": code_context,
    }
    if patch_first:
        payload["edit_guidance"] = (
            "Plan edits before writing code; prefer SEARCH/REPLACE or small targeted hunks; "
            "keep signatures and public exports intact."
        )
    return payload


def plugin_json_dumps(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
