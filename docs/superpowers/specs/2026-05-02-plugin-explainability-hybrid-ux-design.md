# Plugin Explainability — Hybrid Payload + Sidecar (Design Spec)

**Status:** Approved (brainstorm lock-in)  
**Date:** 2026-05-02  
**Scope:** token-reducer plugin / Claude context output — **human trust** and **model nudge** without breaking the **token-reduction** thesis.

**Persona:** Target **C** (simple default + power escape hatches), **optimize for B** (marketplace / unknown skill): one obvious path, no required tuning.

**Related**

- `docs/superpowers/specs/2026-05-02-audit-spine-design.md` — spine is the **only** source for summaries; this spec defines **how** it surfaces to user vs model.
- `docs/superpowers/specs/2026-05-02-context-intelligence-layer-design.md` — confidence/risk semantics must stay compatible where referenced.

---

## 1. Problem statement

Retrieval and compression run, but users do not **feel** why chunks stayed or left, or how hard compression ran—unless they dig through engineer-facing logs.

This spec defines a **strict hybrid**:

- **Minimal model-facing line** (~15 tokens): directional bias, almost free.
- **Short human narrative** (sidecar): trust without friction.
- **Verbose**: counts, categories, examples, full spine — never default.

---

## 2. Non-negotiables

1. **Default explainability is always ON** (for B); verbosity tiers control **depth**, not presence.
2. **Payload summary line is deterministic** — rendered from `audit_spine` (+ stable projection rules). **Never** LLM-generated for v1.
3. **No full explainability block inside model context** — contradicts core product (token waste + signal dilution).
4. **Pure sidecar-only is insufficient** — keep exactly **one** model-facing line derived from spine.

---

## 3. Hybrid surfaces

### 3.1 Payload (model-facing) — **one line only**

Single sentence template (shape, not literal copy):

> Context optimized (`<preset>`, `<confidence band>`). Key focus: `<short focus phrase>`.

**Rules**

- `<preset>`: maps from routing/strategy preset name (e.g. balanced / fast / max — exact enum TBD at implementation).
- `<confidence band>`: derived from spine `outcome.final_confidence` + disagreement/risk flags → discrete bucket (e.g. low / medium / high); thresholds versioned.
- `<short focus phrase>`: **≤12 words**, from spine-backed signals only (e.g. top selected paths / intent bucket hash label / strategy id slug — **no** raw prompt dump).

**Hard cap:** target ~15 tokens; implementation enforces max character budget.

### 3.2 Sidecar (user-facing narrative) — **3–6 lines default**

Structured micro-blocks (emoji optional in UI; plain text in JSON):

```
Selected:
- <path or label> (<one reason>)
- <path> (<one reason>)

Dropped:
- <path glob or label> (<one bucket reason>)
- <path> (<one bucket reason>)

Compression:
<preset> • <confidence band>
```

**Rules**

- **3–6 lines total** including headers — if content overflows, **truncate** lowest-priority drops first; never spill counts into default.
- **One reason per row** — noun phrases, not debug tables.
- **Bucket vocabulary** (closed set): e.g. `low relevance`, `redundant`, `strategy prune`, `tier skip`, `below floor` — mapped from spine + pipeline stage ids.

### 3.3 Verbose (`--verbose` / env)

Everything heavier lives here:

- per-category **counts**
- optional **examples** (extra paths)
- **full `audit_spine` JSON** (or normalized projection)

---

## 4. Data flow

1. Pipeline completes → `audit_spine` populated (see audit spine spec).
2. **Control plane** builds **`signal_snapshot_v1`**, **`gate_decision_v1`**, **`decision_trace_v2`** from the same spine + `ContextRunState` (`emit_decision_trace` / `build_trace_bundle_from_run`). This is the **canonical** reviewer JSON.
3. **Renderer** (hybrid UX): `spine → { payload_one_liner, sidecar_lines[], verbose_bundle }` — **projections only**; must not invent facts not derivable from spine + the same inputs used for `decision_trace_v2`.
4. Plugin payload assembly: inject **only** one-liner into model context path; attach sidecar + spine pointer under stable keys. **`claude_context["decision_trace"]`** carries the full **`decision_trace_v2`** object (same as on-disk `decision_trace.json`). **`ContextPacket.decision_trace`** mirrors it for structured consumers.

**Presentation bridge (reviewer / GTM):** Query → **DecisionTrace** (`decision_trace_v2`) → chunk refs (`selected_chunks` / paths) → model answer / compressed bullets. **`summary.txt`** in the trace directory is template-filled from bundle fields (aligned with the trace, not a second narrative).

### 4.1 Legacy packet fields

`focus_line`, `agent_trace`, and `chunk_transparency` on **`ContextPacket`** are **deprecated** for default packets. Set **`TOKEN_REDUCER_LEGACY_PACKET_EXPLAIN=1`** to populate them (previous behavior). The canonical trust surface is **`decision_trace`** + hybrid one-liner / sidecar.

---

## 5. UX / command shape (Section 3 lock-in)

- **One primary command** — happy path, minimal args.
- **Presets optional** — default = marketplace-safe **balanced**.
- **First-run moment** — user sees sidecar narrative immediately; advanced knobs documented behind env/flags.

(No change from approved brainstorm Section 3.)

---

## 6. Testing / acceptance

- **Golden tests**: same spine snapshot → **byte-stable** one-liner + sidecar (normalization rules documented).
- **Regression**: enabling hybrid explainability does **not** increase model-context bullet payload beyond the one-liner budget (plus existing compressed context).
- **Privacy**: sidecar and one-liner obey same redaction profile as spine (`query.raw` policy).

---

## 7. Explicit non-goals (v1)

- LLM-paraphrased explanations on the hot path.
- Full narrative inside model-facing context.
- Default sidecar longer than **6 lines**.

---

## Document history

| Rev | Notes |
|-----|--------|
| 0.1 | Hybrid strict: 1-line payload, 3–6 line sidecar, verbose for spine + counts; persona B-optimized. |
| 0.2 | Align with **`decision_trace_v2`** as canonical artifact; legacy packet explain fields behind env; presentation bridge + `summary.txt` alignment. |
