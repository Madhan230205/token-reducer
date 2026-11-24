# Intelligence pipeline (3 layers)

This doc describes the **refactored execution model**: classify → route → compress. No extra features—only tighter defaults.

## Folder layout (high level)

| Area | Role |
|------|------|
| `scripts/token_reducer/execution_route.py` | Tier, gates, skill pick, **RoutingPlan** (output cap + subagent profile) |
| `scripts/token_reducer/token_intelligence.py` | Layer 3: dedupe / structured cleanup after compression |
| `scripts/token_reducer/orchestrator.py` | Stage order; honors `ExecutionRoute` |
| `scripts/token_reducer/pipeline.py` | Cache key includes tier; attaches route to state |
| `hooks/pre_processor.py` | Single **UserPromptSubmit** entry (guard + hygiene) |
| `hooks/post_processor.py` | Passthrough for future stop/session hooks |
| `hooks/prompt_guard_lib.py` | Shared guard logic (also used by legacy `userprompt_guard.py`) |
| `skills/skill_registry.json` | Atomic skill metadata (`cost_estimate`, `latency_ms`, `success_rate`) |
| `src/lib.rs` | Rust: token estimate, tokenize, chunking, **collapse_blank_lines**, **trim_to_max_words** |

## Layer 1 — intent classifier

**Output:** `simple` \| `tool` \| `complex`

- Regex / keyword cues (slash commands, navigation, “step by step”, multi-file paths).
- Reuses structured hints from `intent.detect_intent` (`chat` → strongly **simple**).
- Optional override: `TOKEN_REDUCER_TIER_OVERRIDE=simple|tool|complex` (no LLM cost).

## Layer 2 — execution router

| Tier | Subagents | LSP expansion | Retrieval scale | Notes |
|------|-----------|---------------|-----------------|-------|
| **simple** | skipped | skipped | ~0.52× | Default smallest path for short/chat prompts |
| **tool** | skipped | on | ~0.72× | Exactly one skill id from registry efficiency × trigger overlap |
| **complex** | full pipeline | on | 1.0× | Planner-style retrieval / multi-agent pre-compression |

**Reducer / heavy compression:** `expand_context_if_needed` and high compressor modes run only when estimated selected-chunk tokens ≥ `reducer_token_threshold` (default **2800**, env `TOKEN_REDUCER_REDUCER_THRESHOLD`). Below that: **`compression_level=low`** and no expansion pass.

## Layer 3 — token intelligence

Post-compression pass: **near-duplicate bullet removal**, **blank-line collapse** (Rust when built), **summarize-overflow** for oversized bullets (structured fold, not arbitrary tail truncation).

## Sample flows

### SIMPLE

1. Classifier → `simple`.
2. Retrieval caps scaled down; neighborhood expansion skipped.
3. Subagents skipped; LSP skipped.
4. If under token threshold → light compression only; `refine_output_bullets`.

### TOOL

1. Classifier → `tool`; registry picks `skill_id` (efficiency score).
2. Subagents skipped (no multi-agent reducer chain).
3. LSP + retrieval moderate scale; reducer only if chunk mass exceeds threshold.

### COMPLEX

1. Classifier → `complex`.
2. Full retrieval policy; neighborhood expansion when plan allows.
3. Full subagent chain; LSP per policy.
4. Heavy compression + expansion when above threshold.

## Performance impact (qualitative)

- **Latency:** SIMPLE/TOOL avoid subagent passes and often LSP → fewer subprocess/sync steps per turn.
- **Tokens:** Smaller retrieval caps + skipping expansion/reducer on small selections cuts outbound context.
- **Predictability:** Single hook entry (`pre_processor`), fixed orchestrator order, tier in cache key avoids stale reuse across paths.

## Pseudocode

```
tier = classify(prompt, structured_intent)   # regex + heuristics + env override
route = route_table[tier]                    # scale, gates, skill_id

if tier == SIMPLE:
    run retrieval(scale)
    skip neighborhood, subagents, LSP
elif tier == TOOL:
    skill_id = pick_skill(registry, prompt)
    run retrieval(scale)
    skip subagents
else:
    full pipeline

pre_tokens = sum(chunk.token_estimate for chunk in selected)
heavy = pre_tokens >= route.reducer_token_threshold
compress(level = heavy ? intent.level : LOW)
if heavy: maybe_expand_context()

bullets = refine_output_bullets(bullets)     # dedupe + collapse + fold
clamp_to_budget(bullets)
```
