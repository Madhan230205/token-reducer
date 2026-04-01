# token-reducer

Save tokens. Save money. Keep answers sharp.

`token-reducer` is a free, local-first Claude Code plugin that reduces context bloat by sending only high-signal compressed context to the model.

GitHub: https://github.com/Madhan230205/token-reducer

## What it does

1. Preprocesses noisy input
2. Indexes content with SQLite FTS5 + local embeddings
3. Retrieves with BM25-first hybrid logic
4. Reranks and keeps top 3-5 chunks
5. Compresses output into a context packet

Default flow:

`Query → FTS(BM25) → (Vector fallback if needed) → Merge → Top 5 → Compress`

## Why this helps millions of users

- Lower token usage on long tasks
- Better focus for coding/debugging prompts
- Free local core (no required paid embedding API)
- Works in offline-friendly environments

## Repository structure

```text
token-reducer/
├── .claude-plugin/plugin.json
├── .mcp.json
├── .env.example
├── settings.json
├── requirements-optional.txt
├── scripts/
├── hooks/
├── commands/
├── agents/
├── skills/
└── evals/
```

## Local dev use

- Run Claude Code with this plugin directory.
- Use `/token-reducer` command for context-slim retrieval.
- Optional ML+ANN acceleration: install `requirements-optional.txt`.

## Marketplace publish + install

1. Push this folder as a public GitHub repo:
   - `https://github.com/Madhan230205/token-reducer`
2. Register marketplace in Claude Code:
   - `/plugin marketplace add Madhan230205/token-reducer`
3. Install plugin:
   - `claude plugin install token-reducer@madhan230205-marketplace`

For teams:

- `claude plugin install token-reducer@madhan230205-marketplace --scope project`

Detailed rollout checklist: `MARKETPLACE.md`

## Security notes

- `.env` is ignored in git.
- Keep secrets in local env or secret manager.
- Rotate keys immediately if exposed.
