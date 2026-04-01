# Context7 Integration

Use Context7 only when the user asks for external library/API freshness.

1. Resolve library ID.
2. Query docs with precise question.
3. Merge docs snippets into local candidates.
4. Re-rank and compress with same top-5 and token constraints.

If Context7 is unavailable, continue local-only flow.
