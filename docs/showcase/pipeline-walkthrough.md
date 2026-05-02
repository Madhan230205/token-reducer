# Pipeline walkthrough — one real-shaped example

**Flow:** Input → Retrieval → Compression → **Audit Spine** → **Trace bundle** → Output

Numbers in §2 are taken from the published **[BENCHMARK.md](../../BENCHMARK.md)** row for the query *“How does authentication work?”* (same codebase corpus, hash embeddings, FTS-first). Latency/tokens are **reported benchmark results**, not simulated.

---

## 1. Visual path

```mermaid
flowchart LR
  Q[User query] --> I[Intent + route]
  I --> R[Retrieve FTS / vector]
  R --> M[Merge + rerank]
  M --> S[Select + subagents]
  S --> C[Compress + plugin payload]
  C --> SP[Audit spine append]
  SP --> TB[Trace bundle siblings]
  TB --> O[Context packet + JSON]
```

**Sibling artifacts (same `trace_id`):**

- **Signal snapshot** — fused confidence, risk, disagreement, weak retrieval, stats.  
- **Gate decision** — what was *allowed* (vector, retry, compression cap, cache, explainability tier).  
- **Decision trace** — public “gold” kept/dropped/path/compression headline.  
- **Audit spine** — full append-only stage record (does **not** embed the gate).

---

## 2. Example row (authentication query)

| Stage | What happened | Tokens / time (benchmark) |
|-------|----------------|-----------------------------|
| **Input** | Query text + indexed corpus | Raw pool ~34.7k tokens / corpus (see BENCHMARK §1.2) |
| **Retrieval** | FTS hits dominant; vectors optional | **7** chunks surfaced for this query; **17.26 ms** latency |
| **Merge / rank** | Hybrid merge → scored pool | Top chunk scores drive order |
| **Select + compress** | Budget + relevance floor | **~1,597** “input” tokens → **~416** output tokens (**~3.84×** compression on that query) |
| **Spine** | Query, retrieval, routing, compression, outcome slices filled | Single `trace_id` |
| **Trace** | `decision_trace_v1` projection | See [decision-trace-star.md](decision-trace-star.md) |
| **Output** | `ContextPacket` + plugin JSON (`code_context`, hybrid explainability fields) | Delivered to Claude / CLI |

---

## 3. Decisions called out (conceptual)

1. **Route** — execution tier (simple / tool / complex) caps cost before retrieval expands.  
2. **Retrieve** — FTS-first; vector when policy allows and recall needs it.  
3. **Disagreement** — FTS vs vector overlap computed **once** in the spine retrieval slice; downstream reads flags instead of recomputing.  
4. **Compress** — Level and floor adjusted by intent + policy + (bounded) feedback; **hybrid explainability** exposes a **one-line** model cue + **short sidecar**.  
5. **Trace bundle** — Same `trace_id` links spine (facts) to gate (policy) and trace (narrative).

---

## 4. Honest scope note

The diagram is **architecture-accurate**; **full automation** of every label in `decision_trace_v1.path` and sibling population is still being wired from live state. The **schemas and spine** are in-repo; the **star narrative** is this document plus [decision-trace-star.md](decision-trace-star.md).
