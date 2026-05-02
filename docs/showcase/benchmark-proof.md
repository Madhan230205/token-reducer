# Benchmark proof — before / after, retention, disagreement

**Primary source:** [BENCHMARK.md](../../BENCHMARK.md) (Token Reducer **v1.4.0**, April 2026, Windows / Python 3.13, hash embedding backend).

This page is a **proof-oriented snapshot** for reviewers: numbers first, then how **disagreement** fits the audit spine.

---

## 1. Before vs after token usage (corpus-level)

From BENCHMARK **§2.3**:

| | Tokens |
|---|--------|
| **Raw context (indexed corpus)** | **34,671** |
| **Average compressed per query** | **372** |
| **Estimated savings** | **98.9%** |

**Interpretation:** “Before” is the indexed pool scale; “after” is what the pipeline typically sends **per query** after retrieval + compression under the documented settings (see BENCHMARK §1.1).

---

## 2. Query-level before / after (one row)

Query: **“How does authentication work?”**

| | Value |
|---|--------|
| Latency | **17.26 ms** |
| Chunks selected | **7** |
| Input tokens (selected window) | **1,597** |
| Output tokens (compressed) | **416** |
| Compression ratio | **3.84×** |

---

## 3. Accuracy / relevance retention (how we talk about it honestly)

The harness and benchmark docs emphasize **deterministic checks** where scenarios define expected paths/symbols — not a subjective “the answer felt right.”

**What we claim today:**

- Strong **token reduction** with **stable relevance proxies** (BM25 + optional vectors, rerank, structural signals).
- Published **query-level** chunk counts and compression ratios.

**What we do not claim from BENCHMARK.md alone:**

- A single universal “accuracy percentage” against all possible tasks.

**Bridge to ongoing work:** [Benchmark & proof harness spec](../superpowers/specs/2026-05-02-benchmark-proof-harness-design.md) describes recall@k, stability hashes, and regression lanes — the right place to tighten **accuracy retention** as a **numeric gate**.

---

## 4. Disagreement handling (example)

When both FTS and vector channels contribute, the **audit spine** stores disagreement **once** in `retrieval.disagreement`:

```json
{
  "fts_vector_overlap": 0.24,
  "rank_correlation": 0.41,
  "low_overlap_flag": true,
  "mismatch_flag": true
}
```

**Effect chain (illustrative):**

1. **Routing / risk** — `routing.risk_flags` can include `low_overlap` / `high_disagreement` (see audit spine spec).  
2. **Outcome confidence** — `outcome.final_confidence` may be **discounted** when flags fire (see implementation in `explainability_render.confidence_band`).  
3. **Decision trace** — `compression.flags` may include `disagreement_present` when projecting to `decision_trace_v1`.  
4. **UX** — Hybrid sidecar stays short; verbose mode exposes full spine JSON for debugging.

This is **deterministic** signal handling — no LLM-generated “explanation” on the hot path.

---

## 5. Reproduce

See root **[BENCHMARK.md](../../BENCHMARK.md)** for exact CLI / corpus / versions. For plugin-shaped output with hybrid explainability fields, run the current pipeline on a workspace and inspect `context_optimization_line` and `context_explainability` in the plugin JSON.
