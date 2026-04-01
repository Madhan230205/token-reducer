# token-reducer Implementation Guide

## Core design

- FTS first (BM25) for lexical precision.
- Hybrid semantic retrieval for recall when lexical hits are weak.
- ANN acceleration when optional dependencies are available.
- Strict top-3 to top-5 cap before compression.

## Cost controls

- Oversized query guardrails block raw prompt dumps.
- Compression efficiency cap prevents verbose summaries.
- Query cache and embedding cache reduce repeated compute.
- Session memory continuity avoids re-sending large histories.
