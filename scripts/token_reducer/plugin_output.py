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
) -> dict[str, object]:
    summary, relationships = summarize_context(query, intent, candidates)
    files = sorted({c.source for c in candidates})
    code_context: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for c in candidates:
        meta = infer_chunk_meta(c.text, c.source)
        key = (c.source, str(meta.get("symbol_name", "")))
        if key in seen:
            continue
        seen.add(key)
        code = denoise_code_for_plugin(c.text, query)
        if len(code) > 12000:
            code = code[:12000] + "\n…"
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
    return {
        "intent": intent,
        "summary": summary,
        "relevant_files": files,
        "relationships": relationships,
        "code_context": code_context,
    }


def plugin_json_dumps(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
