# Why Token Reducer is different

**One page.** Three ideas that separate this project from “another RAG wrapper.”

1. **Decision-aware compression** — The pipeline does not only shrink text; it **chooses** what to keep using routing, retrieval fusion, and budgets. Those choices are moving toward a single **audit spine** and **sibling control-plane artifacts** (signal snapshot, gate decision, public **decision trace**) so *policy*, *signals*, and *outcome* stay separable and replayable.

2. **Explainability without token cost** — A **strict hybrid** model: one **deterministic, spine-derived line** in model context for directional bias, plus a **short sidecar** (3–6 lines) for human trust. The full story — counts, categories, and raw spine JSON — lives in **verbose** / tooling, not in the answer stream.

3. **Replayable reasoning (audit spine + trace bundle)** — The same `trace_id` links **what happened** (spine), **what was allowed** (gate), **what we sensed** (signal snapshot), and the **reviewer-facing trace**. That is the shape serious systems use to answer “why did it do that?” without conflating outcome with policy.

---

## Honest take

**What is strong today:** local-first retrieval + compression, published benchmark numbers on a real corpus, and the start of a **credible trust surface** (spine + hybrid explainability + typed TraceBundle skeleton).

**What is still in flight:** wiring the control plane so **every** stage consults a single **gate** (no scattered heuristics), and making **decision_trace_v2** the default **star artifact** in docs, harness output, and plugin UX.

**What this doc is not:** a claim that shipping is “finished.” It is a **positioning** anchor: if you are comparing tools, these three properties are the bar Token Reducer is building toward.

For proof-oriented numbers, see [showcase/benchmark-proof.md](showcase/benchmark-proof.md) and the full [BENCHMARK.md](../BENCHMARK.md) in the repo root.
