# Benchmark & Proof Harness — Design Spec

**Status:** Approved for implementation planning (rev 2 — decision trace, regression thresholds, stability hash, external pin criteria, tags)  
**Date:** 2026-05-02  
**Scope:** token-reducer — reproducible benchmarking, stage attribution, deterministic quality gates, optional non-authoritative LLM judging, and published evidence (`BENCHMARK.md` + CI artifacts).

---

## 1. Purpose

Ship **hard evidence** that the pipeline improves or preserves retrieval usefulness while controlling cost (latency and tokens). Without this layer, adaptive routing, context intelligence, and “magic” narratives remain unfalsifiable.

**Primary outcomes**

- **Ground truth for regressions:** same scenario inputs yield comparable metrics across commits.
- **Bottleneck visibility:** per-phase latency and token counters attributed to existing orchestration seams.
- **Credibility:** reviewers can reproduce runs using pinned fixtures and (where enabled) a pinned external repository SHA.
- **Feedback surface for Phase B:** artifact schema exports cohort-level failures and stage regressions for a future adaptive control plane (out of scope here).

**Non-goals (v1)**

- Training or online reinforcement of routing or skills inside this harness.
- Replacing deterministic CI with LLM judge scores as merge gates.
- Submodule vendoring of third-party repositories.
- Proving subjective “user delight”; weekly judge output is explicitly advisory.

**Roadmap note:** **Phase D (this spec) precedes Phase B (adaptive control plane).** Phase B should consume exported artifacts; it is not specified beyond interface hints in Section 9.

---

## 2. Accuracy policy

**Primary:** deterministic checks only for merge-blocking lanes.

- Assertions use scenario-defined **`expected_paths`** (repository-relative, normalized) and optional future extensions (`expected_symbols`, etc.).
- Report **recall@k** against expected paths where k aligns with selection caps documented per scenario.
- Record **stability hashes** for normalized outputs when scenarios require byte-stable bullets or payload subsets (normalization rules: Section 6.2).

**Secondary (scheduled): LLM-as-judge**

- Runs **weekly** (or manual dispatch), **never** blocks PR merge unless explicitly promoted later.
- Inputs are **minimized:** scenario rubric plus post-pipeline context payload (e.g., compressed bullets / plugin summary), not arbitrary workspace dumps.
- Every judge row includes `method: llm_judge`, model id, prompt template version, and a **non-authoritative disclaimer**.
- Judge timeouts or parse failures **soft-fail:** record error row; do not fail deterministic tiers.

---

## 3. Scenario model

Each scenario is a stable **`scenario_id`** with:

| Field | Description |
|-------|-------------|
| `tier` | `smoke` \| `nightly` \| `weekly` |
| `fixture_id` | In-repo path under `benchmarks/fixtures/` or external pin reference |
| `query` | Frozen query text |
| `env` | Frozen knobs (routing flags, embedding backend overrides where applicable) |
| `expected_paths` | List of paths that must appear in selected sources for pass (deterministic) |
| `required` | If true, participates in **PR-blocking** subset when tier includes smoke |
| `tags` (optional) | Free-form labels for cohort analysis (e.g. `deep_retrieval`, `compression_sensitive`); ignored by pass/fail unless a workflow explicitly filters on them |

**Promotion:** new scenarios default `required: false`; after stability review (or N clean nightly runs), maintainers set `required: true`.

---

## 4. CI tiers and gates

### 4.1 Smoke (PR CI)

- Runs **in-repo micro-fixtures only** — **no network fetch** by default.
- **`required: true`** failures **fail** the job.
- **`required: false`** failures emit **warnings** visible in CI summary (non-blocking).

### 4.2 Nightly

- Smoke scenarios plus **`nightly`**-tagged scenarios.
- May clone **one allowlisted external repository** at a **pinned full SHA** when `TOKEN_REDUCER_BENCHMARK_FETCH=1` (or CI equivalent).
- Deterministic failures **fail** the nightly workflow for scenarios in the nightly set (accountability without freezing PR throughput).

### 4.3 Weekly

- Runs optional **LLM judge** lane on a subset of scenarios; outputs appended to artifacts; **non-blocking**.

### 4.4 Skip semantics

If external fetch is disabled but a scenario requires it, mark run as **`skipped: network_disabled`** — never silently pass.

### 4.5 Regression thresholds (default policy)

Metrics are compared to a **stored baseline** (e.g. last green `main` artifact for the same `scenario_id` and tier). When no baseline exists, record metrics only — do not fail on regression until a baseline is seeded.

| Signal | Default regression (warning or fail per workflow config) | Notes |
|--------|-----------------------------------------------------------|--------|
| Total scenario wall latency | **+10%** vs baseline | Compare end-to-end runner time for that scenario unless a stage-level regression is specified later |
| Token totals (harness-defined rollup field, e.g. post-compression estimate) | **+15%** vs baseline | Same field must be used across runs |
| Recall@k (deterministic path coverage) | **−5 percentage points** vs baseline (e.g. 0.85 → 0.80) | Absolute drop, not relative |

**PR clarity:** smoke **`required`** scenarios may treat regression thresholds as **warnings** initially; nightly may promote the same thresholds to **fail** once baselines stabilize. Exact wiring is implementation-specific but defaults above are normative for interpreting artifacts.

---

## 5. Repository layout

```
benchmarks/
  fixtures/           # synthetic micro-workspaces (smoke-heavy)
  scenarios/          # scenario definitions (YAML or JSON)
  external_pins.toml  # exactly one canonical external URL + full commit SHA for v1
```

**Hybrid fixture strategy**

- **In-repo:** default for PR speed and air-gapped reproducibility.
- **External:** shallow clone at pinned SHA into `benchmarks/_download/<sha>/` with **host/path allowlist** enforcement.

**Index cache**

- Store under `benchmarks/_index_cache/` locally; CI uses cache keyed by `(pinned_sha, indexer_config_hash, embedding_parameters)`.

### 5.1 External pin selection criteria (v1)

The single canonical external repository MUST be:

- **Medium-sized:** on the order of **1k–5k indexed files** at the pinned SHA (avoid tiny toys and million-file monorepos for v1).
- **Clear module boundaries:** package or src layout that makes **expected_paths** assertions meaningful.
- **Backend-leaning:** aligns with typical token-reducer usage (e.g. Python/FastAPI-style services); React-sized frontends remain valid later as additional pins if the project expands beyond one external SHA.

The implementation plan MUST record the chosen URL, **full** pinned SHA, LICENSE name, and approximate file count at pin time.

---

## 6. Metric contract (runner output)

Each scenario emits **one JSON Lines record** (schema versioned, start at **`benchmark_proof_v1`**) containing at minimum:

| Field group | Content |
|-------------|---------|
| Identity | `scenario_id`, `tier`, `required`, optional `tags`, token-reducer git SHA, schema version |
| Timing | Wall milliseconds per top-level phase (cost prepare, retrieve, reasoning, compress) and substeps where cheap |
| Tokens | Estimates at selection boundary, post-subagents, post-compression (as available from pipeline) |
| Retrieval | Pool sizes, retry fired, vector path, sparse/weak flags surfaced by existing pipeline debug |
| **Decision trace** | Routing and orchestration snapshot (see below); **required for blaming retrieval vs routing** |
| Deterministic result | pass/fail, missing paths, recall@k |
| Regression flags | Booleans or severity vs baseline per Section 4.5 when baseline present |
| Optional judge | rubric scores, rationale text, disclaimer |

### 6.1 Decision trace (required field group)

Every scenario record MUST include `decision_trace`, populated from existing runtime state / debug payloads (`execution_route`, `routing_plan`, `subagent_debug`, multi-agent trace, etc.). Exact wiring is implementation-defined; semantics are fixed:

```json
"decision_trace": {
  "tier": "simple|tool|complex",
  "skill_selected": "string|null",
  "subagents_used": ["filter", "rank"],
  "compression_triggered": true
}
```

| Key | Meaning |
|-----|---------|
| `tier` | Final **`ExecutionRoute.tier`** (`simple`, `tool`, or `complex`). |
| `skill_selected` | When **`tier == tool`**, the chosen skill id/name from routing if present; otherwise `null`. |
| `subagents_used` | Ordered list of **chunk-level subagent steps** from `subagent_debug` (e.g. filter, rank, fuse). If subagents were skipped, list empty or a single sentinel documented in the implementation plan. |
| `compression_triggered` | `true` iff the compressor specialist phase ran and produced output for that scenario run (not merely scheduled). |

Top-level specialist phases (cost optimizer, retriever, …) remain in the **Timing** field group; `decision_trace` focuses on **tier/skill/subagent/compression** blame lines needed for Phase B.

### 6.2 Stability hash for normalized outputs

When a scenario enables output hashing, compute **SHA-256** (or same digest as elsewhere in the repo) over a **canonical UTF-8 string** built as follows:

1. **Selected sources:** sort paths lexicographically, normalize to forward slashes, strip leading `./`, lowercase drive letters on Windows if applicable.
2. **Bullets:** split final bullet list into lines; trim trailing whitespace per line; normalize line endings to `\n`; ensure **stable order** (scenario specifies whether order-sensitive or sort lexicographically after strip).
3. **Whitespace:** collapse internal runs of spaces/tabs within each line to single space where the scenario marks content as whitespace-insensitive; otherwise preserve intentional indentation blocks per scenario rubric.
4. **Plugin payload subset (optional):** if hashed, scenario lists exact JSON keys; serialize with **sorted keys**, no insignificant floats.

Record in the artifact: `stability_hash_algorithm`, `stability_hash_inputs` (which bullet modes / keys), and the digest. Prevents flaky CI from nondeterministic ordering or OS-specific paths.

---

## 7. Publishing `BENCHMARK.md`

**On merge to `main`**

- Regenerate headline tables from **smoke tier, `required` scenarios only** for fast, gate-aligned numbers.

**Nightly**

- Extend generated markdown (or companion section) with nightly-only rows: external pin id, **pinned SHA**, run timestamp, broader scenario coverage.

**PR branches**

- Do **not** overwrite canonical tracked `BENCHMARK.md` from arbitrary PR workflows; optional PR comments or artifact uploads only.

**Automation mechanics**

- A workflow opens a **docs-only pull request** against `main` containing regenerated benchmark markdown (and any generated appendix). Humans merge after review.

**Integrity:** `BENCHMARK.md` (and generated appendices) MUST be produced **only** from pinned artifact JSON/schema versions — never hand-edited headline numbers. This preserves reproducibility and auditability.

---

## 8. Components

| Unit | Responsibility |
|------|----------------|
| Loader | Validate scenario files, enforce allowlists, resolve fixture paths |
| Runner | Invoke existing pipeline entrypoints with frozen env; capture metrics |
| Reporter | Emit JSON Lines + markdown tables + GitHub summary snippets |
| Fetcher | Shallow clone external pin with SHA verification |
| Judge adapter | Weekly-only HTTP client; secrets via CI; strict timeouts |

**Errors**

- Schema mismatch or runner crash in deterministic tiers → **fail** the owning workflow tier.
- Judge lane errors → **record** and continue.

---

## 9. Phase B interface (informational)

Exported artifacts SHOULD include aggregates suitable for adaptive policy without prescribing algorithms:

- Per-scenario failure counts by failure class (missed path, retry divergence, latency regression).
- Stage-level p95 deltas vs rolling baseline window (computed in CI or offline tooling).
- **`decision_trace` cohorts:** failure rates conditional on `tier`, `skill_selected`, and `subagents_used` patterns (Phase B input).

No storage format beyond JSON Lines aggregate exports is mandated in v1.

---

## 10. Security and privacy

- External clone URLs must match an explicit **allowlist**.
- Judge credentials are **CI secrets**; logs redact tokens.
- Document that weekly judging transmits **only** minimized payloads needed for rubric scoring.

---

## 11. Testing strategy

- Unit tests for loader validation, path normalization, reporter determinism.
- Golden-string tests for markdown fragments generated from fixed fixture JSON inputs.
- Tests that **`decision_trace`** schema is present and stable for a mocked pipeline run.
- Tests for **stability hash** canonicalization (path sort order, bullet normalization modes).
- At least one smoke scenario executes the **real indexer path** on a micro-fixture.

---

## 12. Self-review checklist

| Check | Result |
|-------|--------|
| Deterministic vs judge precedence | Deterministic gates merge; judge advisory |
| Hybrid fixtures | In-repo default; one pinned external SHA behind fetch flag |
| PR policy | Required blocks; experimental warns |
| External pin criteria | Medium size, module boundaries, backend-leaning v1 pin documented in plan |
| Decision trace | Tier, skill, subagents, compression in every record |
| Regression thresholds | Default +10% latency, +15% tokens, −5pp recall vs baseline |
| Stability hash | Canonical paths, bullets, optional payload keys defined |
| Scenario tags | Optional cohort labels supported |
| Publishing | Main reflects required smoke; nightly extends; docs via PR; no hand-edited numbers |
| Scope | Benchmark harness only; adaptive learning deferred |

---

## 13. Next step

Use the **writing-plans** workflow to produce an implementation plan (runner wiring, schema version constant, initial scenario set, CI workflows, docs PR bot).
