# Phase 1: Performance & Architecture Overhaul
> token-reducer — LLM-executable plan for consecutive chat sessions
> Last updated: 2026-04-04

---

## Phase 0: Documentation Discovery (READ BEFORE ANY PHASE)

**Goal:** Ground every implementation in actual API docs before writing a line.

### 0A — PyO3 / Maturin docs

In each phase chat session, fetch these via Context7 before touching any Rust/Python boundary code:

| Topic | Context7 query |
|-------|----------------|
| Maturin project init | `resolve-library-id: "maturin"` then `query-docs: "new project mixed layout pyproject.toml"` |
| PyO3 function export | `resolve-library-id: "pyo3"` then `query-docs: "pyfunction module init wrap rust function python"` |
| PyO3 accepting Python str/bytes | `query-docs: "PyString PyBytes extract str"` |
| PyO3 returning Vec<f32> to Python list | `query-docs: "PyList IntoPy Vec return type"` |
| Maturin develop (editable install) | `query-docs: "maturin develop --release"` |
| Maturin CI wheel build | `query-docs: "maturin build --release wheel github actions"` |

**Anti-patterns confirmed from docs (do NOT invent these):**
- Do NOT use `#[pyclass]` on plain structs unless they need to cross the boundary as objects — prefer returning primitive Python types (str, list, int, float) from `#[pyfunction]`
- Do NOT use `pyo3::Python::with_gil` in tight loops — acquire GIL once per boundary call
- Do NOT add `cdylib` to `lib.rs` manually — Maturin's `pyproject.toml` handles it

### 0B — FastAPI / MCP server docs

| Topic | Context7 query |
|-------|----------------|
| FastAPI background startup | `resolve-library-id: "fastapi"` then `query-docs: "lifespan startup shutdown background"` |
| FastAPI run in subprocess | `query-docs: "uvicorn run programmatic subprocess daemon"` |
| MCP Python SDK tool registration | `resolve-library-id: "mcp"` then `query-docs: "server tool register FastMCP"` |
| MCP stdio transport | `query-docs: "stdio transport server run"` |

### 0C — uv packaging docs

| Topic | Context7 query |
|-------|----------------|
| uv tool install | `resolve-library-id: "uv"` then `query-docs: "tool install self-contained script"` |
| uv lock reproducible | `query-docs: "lock file reproducible install"` |
| uv build wheel | `query-docs: "build wheel pyproject"` |

---

## Phase 1A — Rust Hot-Path Extraction (PyO3/Maturin)

### Context for the agent

The project lives at `c:\Users\Madv6\Claude_custom_plugins\token-reducer`.
It is a pure Python package (`hatchling` build, no Rust yet).
You are adding a Rust extension crate called `token_reducer_core` that Python imports as a regular module.

**Codebase files you MUST read before making changes:**
- [scripts/token_reducer/chunker.py](scripts/token_reducer/chunker.py) — all of it
- [scripts/token_reducer/embeddings.py](scripts/token_reducer/embeddings.py) — all of it
- [scripts/token_reducer/compressor.py](scripts/token_reducer/compressor.py) — lines 1-120 (TextRank + knapsack)
- [pyproject.toml](pyproject.toml) — build system and extras

### What to implement

#### Step 1 — Scaffold the Maturin crate

1. Read Maturin docs (`query-docs: "mixed layout pyproject.toml"`) to confirm exact config syntax.
2. Create `Cargo.toml` at repo root with `[lib] name = "token_reducer_core" crate-type = ["cdylib"]`.
3. Add to `pyproject.toml` `[build-system]`: replace `hatchling` with `maturin` — **read the Maturin docs** for the exact `requires` and `build-backend` values before editing.
4. Create `src/lib.rs` with the `#[pymodule]` stub (empty module, no functions yet).
5. Verify scaffold: `maturin develop --release` from repo root must produce `token_reducer_core.*.pyd` (Windows) or `.so` (Linux/Mac).

**Verification:** `python -c "import token_reducer_core; print('ok')"` exits 0.

#### Step 2 — Port `tokenize()` and `char_ngrams()`

**Source to port:**
- [scripts/token_reducer/chunker.py:tokenize](scripts/token_reducer/chunker.py) — normalizes text, splits on `\W+`, lowercases, removes stopwords from a short list
- [scripts/token_reducer/chunker.py:char_ngrams](scripts/token_reducer/chunker.py) — generates 3/4-grams from a term string

**Rust implementation notes:**
- Accept `&str`, return `Vec<String>` (PyO3 converts this to `list[str]` automatically)
- Use `regex` crate for `\W+` split (add to Cargo.toml dependencies)
- Stopwords: hardcode the same list currently in `chunker.py` as a `phf_set!` (add `phf` crate)
- Export as `#[pyfunction] fn tokenize(text: &str) -> Vec<String>` and `#[pyfunction] fn char_ngrams(term: &str, min_n: usize, max_n: usize) -> Vec<String>`

**Call-site change in Python** — in `chunker.py`, add at top:
```python
try:
    from token_reducer_core import tokenize as _tokenize_rust, char_ngrams as _char_ngrams_rust
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
```
Then in the body of `tokenize()` and `char_ngrams()`: delegate to Rust if `_RUST_AVAILABLE`, else run existing Python.

**Verification:**
```bash
python -c "from token_reducer_core import tokenize; assert tokenize('Hello World foo!') == tokenize_python_reference('Hello World foo!')"
pytest tests/test_chunker.py::TestTokenize tests/test_chunker.py::TestCharNgrams -v
```

#### Step 3 — Port `estimate_tokens()`

**Source:** [scripts/token_reducer/chunker.py:estimate_tokens](scripts/token_reducer/chunker.py) — `word_count * 1.3`, uses `split()` approximation.

**Rust:**
```rust
#[pyfunction]
fn estimate_tokens(text: &str) -> usize {
    let words = text.split_whitespace().count();
    ((words as f64) * 1.3).ceil() as usize
}
```

**Verification:** `pytest tests/test_chunker.py::TestEstimateTokens -v`

#### Step 4 — Port `embed_text_hash()`

**Source:** [scripts/token_reducer/embeddings.py:embed_text_hash](scripts/token_reducer/embeddings.py)
- Blake2b hash of normalized text → 256-dim float32 vector
- Uses per-token hash projection, accumulates into a float array

**Rust:**
- Add `blake2` and `bytemuck` crates
- Accept `&str`, return `Vec<f32>` (384-dim to match existing ONNX output, or use the current 256 dim — read the Python source first to confirm dimension)
- Export as `#[pyfunction] fn embed_hash(text: &str, dims: usize) -> Vec<f32>`

**Call-site change:** In `embeddings.py:embed_text_hash()`, delegate to `token_reducer_core.embed_hash()` if available.

**Verification:**
```bash
pytest tests/test_embeddings.py::TestEmbedTextHash -v
# Cosine similarity between Python and Rust outputs must be > 0.999
```

#### Step 5 — Port TextRank inner loop

**Source:** [scripts/token_reducer/compressor.py:textrank_score_sentences](scripts/token_reducer/compressor.py) lines ~22-80
- Builds similarity matrix, runs power iteration (30 steps)
- Inner loop is O(n²) per iteration — clear bottleneck for large chunks

**Rust:**
- Accept `Vec<Vec<f32>>` (similarity matrix), `usize` iterations, `f64` damping factor
- Return `Vec<f64>` (final scores)
- Export as `#[pyfunction] fn textrank_iterate(similarity_matrix: Vec<Vec<f32>>, iterations: usize, damping: f64) -> Vec<f64>`
- Use ndarray crate for matrix ops (add to Cargo.toml)

**Call-site change:** In `compressor.py:textrank_score_sentences()`: build similarity matrix in Python (unchanged), pass to `token_reducer_core.textrank_iterate()`.

**Verification:**
```bash
pytest tests/ -k "textrank or compress" -v
# Scores must match Python within 1e-4 tolerance
```

### Anti-patterns for Phase 1A

- Do NOT rewrite `chunk_code()` (tree-sitter) in Rust — tree-sitter has its own Rust crate but Python bindings work fine; don't touch this.
- Do NOT attempt to port ONNX inference — ONNX Runtime has no meaningful overhead vs Rust for batch inference; ML stays in Python.
- Do NOT use `unsafe` blocks unless absolutely required by a crate's API.
- Do NOT change the public Python API signatures of `tokenize()`, `char_ngrams()`, `estimate_tokens()`, `embed_text_hash()`.

### Phase 1A Verification Checklist

- [ ] `maturin develop --release` succeeds with no warnings
- [ ] `python -c "import token_reducer_core"` exits 0
- [ ] All existing tests pass: `pytest tests/ -v`
- [ ] Benchmark: `python scripts/token_reducer/benchmark.py` shows ≥2× speedup on tokenize/embed_hash vs pre-Rust baseline
- [ ] `grep -r "token_reducer_core" scripts/` shows imports only via try/except (graceful fallback)

---

## Phase 1B — Daemonize the Pipeline (MCP Server)

### Context for the agent

Read these before coding:
- [scripts/token_reducer/pipeline.py](scripts/token_reducer/pipeline.py) — `run_retrieval_pipeline()` is what the daemon must expose
- [scripts/token_reducer/cli.py](scripts/token_reducer/cli.py) — `index` command logic (lines 92+) is the second daemon endpoint
- [.mcp.json](.mcp.json) — current MCP config (context7 only); you will add a new entry here
- [pyproject.toml](pyproject.toml) — you will add a new entry point and optional dependency

**Fetch docs first:**
- MCP Python SDK: `resolve-library-id: "mcp"` → `query-docs: "FastMCP server tool stdio"` (get exact import paths and `@mcp.tool` decorator syntax)
- uv run: `query-docs: "uv run script entry point"` (how to make the daemon launchable via `uv run token-reducer-mcp`)

### What to implement

#### Step 1 — New file: `scripts/token_reducer/mcp_server.py`

Create (do NOT modify existing files yet) a new MCP server file:

```python
# scripts/token_reducer/mcp_server.py
# Exposes token-reducer as an MCP tool server.
# Run via: python -m token_reducer.mcp_server
# or: uv run token-reducer-mcp
```

Tools to expose (read MCP docs for `@mcp.tool` exact syntax):
1. `index_files(paths: list[str], db: str, chunk_size: int = DEFAULT_CHUNK_SIZE)` → `str` (JSON stats)
   - Internally calls the same logic as `cli.py:index` command
   - Import from `token_reducer.db` and `token_reducer.chunker` directly
2. `retrieve(query: str, db: str, top_k: int = DEFAULT_TOP_K, word_budget: int = DEFAULT_WORD_BUDGET)` → `str` (JSON ContextPacket)
   - Calls `pipeline.run_retrieval_pipeline()` with sqlite connection
   - Return `packet.model_dump_json()` (Pydantic already defined in models.py)
3. `get_stats(db: str)` → `str` (JSON index stats)
   - Calls `db.get_index_stats()`

Server startup:
```python
if __name__ == "__main__":
    mcp.run(transport="stdio")  # exact syntax from MCP docs
```

**Do NOT use FastAPI** for the initial MCP server — use the MCP Python SDK's stdio transport, which is already the protocol Claude agents use natively. FastAPI would add a network layer that's unnecessary for local use.

#### Step 2 — Register entry point in `pyproject.toml`

Add to `[project.scripts]`:
```toml
token-reducer-mcp = "token_reducer.mcp_server:main"
```

Add `mcp` to `[project.dependencies]` (core, not optional — this is the daemon mode).

#### Step 3 — Add to `.mcp.json`

Append a new server entry (read the existing `.mcp.json` format first):
```json
"token-reducer": {
  "command": "uv",
  "args": ["run", "token-reducer-mcp"],
  "cwd": "<project-root>"
}
```

#### Step 4 — Add `__main__.py` guard to package

In [scripts/token_reducer/__init__.py](scripts/token_reducer/__init__.py), no change needed.
Create `scripts/token_reducer/__main__.py`:
```python
from .cli import main
main()
```
This allows `python -m token_reducer` to work alongside the MCP server.

### Anti-patterns for Phase 1B

- Do NOT spin up a FastAPI/uvicorn server for the MCP daemon — MCP stdio transport is correct for Claude agent integration; HTTP adds latency and a port management problem.
- Do NOT daemonize with `subprocess.Popen` / `nohup` — the MCP host (Claude Code) manages process lifecycle.
- Do NOT duplicate pipeline logic into mcp_server.py — import from existing modules.

### Phase 1B Verification Checklist

- [ ] `python -m token_reducer.mcp_server` starts without errors (Ctrl+C to stop)
- [ ] `mcp list tools` (or MCP inspector) shows `index_files`, `retrieve`, `get_stats`
- [ ] `token-reducer-mcp` is available after `pip install -e .` or `maturin develop`
- [ ] End-to-end MCP test: index a directory via `index_files`, then `retrieve` returns a valid ContextPacket
- [ ] `.mcp.json` updated and committed

---

## Phase 1C — Drop Optionality, Enforce Defaults (uv Packaging)

### Context for the agent

Read these before coding:
- [pyproject.toml](pyproject.toml) — current `[project.optional-dependencies]` (ml, ast, full, dev)
- [requirements-optional.txt](requirements-optional.txt) — onnxruntime, tokenizers, huggingface_hub, tree-sitter parsers
- [scripts/token_reducer/embeddings.py](scripts/token_reducer/embeddings.py) — `DEFAULT_EMBEDDING_BACKEND = "onnx"` (line ~30); ONNX is already the default

**Goal:** Users should run `pip install claude-token-reducer` (or `uv tool install`) and get a fully working tool with ONNX embeddings out of the box — no manual `pip install claude-token-reducer[ml]`.

**Fetch docs first:**
- uv tool: `resolve-library-id: "uv"` → `query-docs: "tool install isolated environment"` (understand uv tool vs pip)
- Maturin wheel: `query-docs: "include data files MANIFEST wheel"` (for bundling ONNX model)

### What to implement

#### Step 1 — Promote core ML deps to required

In `pyproject.toml`, move these from `[project.optional-dependencies].ml` to `[project.dependencies]`:
- `onnxruntime>=1.17.0`
- `numpy>=1.24.0`
- `tokenizers>=0.15.0`  ← needed by ONNX tokenizer
- `huggingface_hub>=0.20.0`  ← for model download on first run

Keep `sentence-transformers`, `hnswlib`, `faiss-cpu` in `[ml]` optional (heavy deps, power users only).
Keep `tree-sitter` and language parsers in `[ast]` optional (AST chunking is a bonus feature).

**Do NOT** remove the `[ml]` and `[ast]` extras — keep them for users who want the heavier backends.

#### Step 2 — Add `uv.lock` and `uv` tooling

1. Run `uv lock` to generate `uv.lock` (reproducible installs).
2. Add `uv.lock` to git (do NOT gitignore it).
3. Update [README or docs] install instructions to show `uv tool install claude-token-reducer` as the primary install method (if docs exist; if not, skip).

#### Step 3 — Verify self-contained install

Test the following sequence in a **fresh virtual environment** (create one with `uv venv .venv-test`):
```bash
uv pip install -e .
token-reducer --help                          # CLI works
token-reducer index tests/ --db /tmp/test.db  # indexing works without [ml] install
token-reducer query "example" --db /tmp/test.db  # ONNX embedding works
```

If the ONNX model download fails (first run, no cache):
- Check `embeddings.py:get_onnx_session()` — it calls `huggingface_hub.hf_hub_download`
- Ensure `huggingface_hub` is now in core deps (Step 1 above)
- First-run download is acceptable; bundling the model binary is out of scope for Phase 1

#### Step 4 — Remove `requirements-optional.txt`

Once the core ML deps are in `pyproject.toml`:
1. Delete `requirements-optional.txt` (it creates confusion about what's needed).
2. `grep -r "requirements-optional" .` to find any references and remove them.
3. Update `.github/workflows/` if it references this file.

### Anti-patterns for Phase 1C

- Do NOT add `sentence-transformers` to core deps — it pulls in PyTorch and is 1GB+. ONNX is the correct default.
- Do NOT bundle the ONNX model binary in the wheel — HuggingFace Hub download on first run is the correct pattern.
- Do NOT use PyInstaller — it creates brittle single-file executables; `uv tool install` is the modern equivalent and works with extension modules.

### Phase 1C Verification Checklist

- [ ] `pip install claude-token-reducer` (from wheel) works without extras
- [ ] `token-reducer index <dir> --db /tmp/x.db && token-reducer query "foo" --db /tmp/x.db` works in fresh env
- [ ] `requirements-optional.txt` deleted; no dead references remain (`grep -r "requirements-optional" .` returns nothing)
- [ ] `uv.lock` committed to repo
- [ ] `pytest tests/ -v` passes with only core deps installed

---

## Execution Order Summary

| Order | Phase | Session focus | Gate to unlock next |
|-------|-------|---------------|---------------------|
| 1 | 1A Step 1 | Maturin scaffold + empty crate | `import token_reducer_core` works |
| 2 | 1A Steps 2-4 | Port tokenize, char_ngrams, estimate_tokens, embed_hash | All chunker/embedding tests pass |
| 3 | 1A Step 5 | Port TextRank inner loop | All compressor tests pass; 2× benchmark improvement |
| 4 | 1B | MCP server + .mcp.json | MCP inspector shows 3 tools; end-to-end test passes |
| 5 | 1C | Promote deps + uv.lock + delete requirements-optional.txt | Fresh env install + full CLI works |

---

## Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| PyO3 version pinning conflicts with other wheels | Pin `pyo3 = "0.22"` in Cargo.toml; check PyO3 changelog before upgrade |
| `token_reducer_core` not importable in CI (no Rust toolchain) | The try/except fallback in each module handles this gracefully |
| MCP SDK API changes | Fetch docs fresh in each session; don't rely on this plan's inline snippets |
| Windows `.pyd` extension vs Linux `.so` | Maturin handles this automatically — do not add platform checks manually |
| HuggingFace Hub first-run download in offline environments | Document `TRANSFORMERS_OFFLINE=1` env var as workaround; out of scope for Phase 1 |
