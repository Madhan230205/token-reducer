#!/usr/bin/env python3
"""A/B eval: does the token-reducer MCP server cut *real* Claude Code token usage?

Runs each task twice through ``claude -p`` in the target repo:
  A (baseline)  — Claude Code with its native Read/Grep/Glob only.
  B (treatment) — same, plus the token-reducer MCP server.
Records total tokens, cost, turns, duration and whether the answer passed the task's check.

Usage:
  python scripts/ab_eval.py --repo /path/to/repo --tasks evals/ab_tasks.sample.json [--repeats 2]
  python scripts/ab_eval.py --repo . --tasks evals/ab_tasks.sample.json --dry-run

Tasks file: [{"id": "...", "prompt": "...", "expect": ["substring", ...]}, ...]
``expect`` is optional; the task passes when every substring appears in the answer (case-insensitive).
Both arms use --strict-mcp-config so only the eval's own MCP servers load. Run from a shell where
token-reducer is NOT already installed as a plugin, otherwise its hook/skill leak into the baseline.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MCP_ENTRY = HERE / "token_reducer_mcp.py"
BASE_TOOLS = ["Read", "Grep", "Glob", "LS"]
MCP_TOOLS = [
    "mcp__token-reducer__search_code",
    "mcp__token-reducer__get_context",
    "mcp__token-reducer__find_symbol",
    "mcp__token-reducer__read_file_slice",
    "mcp__token-reducer__index_status",
]
NUDGE = (
    "This workspace exposes an MCP server named token-reducer. Call search_code first to locate "
    "relevant code, and read_file_slice for exact line ranges, instead of reading whole files or "
    "grepping broadly. Only fall back to Read/Grep when those tools come up empty."
)


def mcp_config(repo: Path) -> dict:
    return {
        "mcpServers": {
            "token-reducer": {
                "command": sys.executable,
                "args": [str(MCP_ENTRY)],
                "env": {"TOKEN_REDUCER_WORKSPACE": str(repo), "PYTHONIOENCODING": "utf-8"},
            }
        }
    }


def build_cmd(
    prompt: str,
    arm: str,
    repo: Path,
    cfg_path: Path | None,
    model: str | None,
    max_turns: int,
    nudge: bool,
) -> list[str]:
    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "json",
        "--max-turns",
        str(max_turns),
        "--strict-mcp-config",
    ]
    tools = list(BASE_TOOLS)
    if arm == "B":
        assert cfg_path is not None
        cmd += ["--mcp-config", str(cfg_path)]
        tools += MCP_TOOLS
        if nudge:
            cmd += ["--append-system-prompt", NUDGE]
    cmd += ["--allowedTools", *tools]
    if model:
        cmd += ["--model", model]
    return cmd


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()


def run_once(cmd: list[str], repo: Path, timeout: int) -> dict:
    exe = shutil.which(cmd[0]) or cmd[0]
    t0 = time.time()
    proc = subprocess.Popen(
        [exe, *cmd[1:]],
        cwd=str(repo),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        with contextlib.suppress(Exception):
            proc.communicate(timeout=15)
        return {"error": "timeout", "wall_s": round(time.time() - t0, 1)}
    wall = round(time.time() - t0, 1)
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "error": f"non-json output (exit {proc.returncode})",
            "stderr": stderr[-800:],
            "stdout": stdout[-800:],
            "wall_s": wall,
        }
    u = data.get("usage") or {}
    total = sum(
        int(u.get(k, 0) or 0)
        for k in (
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "output_tokens",
        )
    )
    return {
        "result": str(data.get("result", "")),
        "tokens_total": total,
        "tokens_uncached_input": int(u.get("input_tokens", 0) or 0),
        "tokens_cache_read": int(u.get("cache_read_input_tokens", 0) or 0),
        "tokens_cache_create": int(u.get("cache_creation_input_tokens", 0) or 0),
        "tokens_output": int(u.get("output_tokens", 0) or 0),
        "cost_usd": float(data.get("total_cost_usd", 0.0) or 0.0),
        "turns": int(data.get("num_turns", 0) or 0),
        "duration_ms": int(data.get("duration_ms", 0) or 0),
        "wall_s": wall,
        "is_error": bool(data.get("is_error", False)),
    }


def passed(result: str, expect: list[str]) -> bool | None:
    if not expect:
        return None
    low = result.lower()
    return all(e.lower() in low for e in expect)


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"n": 0}
    checks = [r["pass"] for r in ok if r.get("pass") is not None]
    return {
        "n": len(ok),
        "errors": len(rows) - len(ok),
        "tokens_total_mean": round(statistics.mean(r["tokens_total"] for r in ok)),
        "tokens_total_median": round(statistics.median(r["tokens_total"] for r in ok)),
        "cost_usd_mean": round(statistics.mean(r["cost_usd"] for r in ok), 4),
        "turns_mean": round(statistics.mean(r["turns"] for r in ok), 2),
        "duration_s_mean": round(statistics.mean(r["duration_ms"] for r in ok) / 1000, 1),
        "pass_rate": round(sum(checks) / len(checks), 3) if checks else None,
    }


def pct(a: float, b: float) -> str:
    if not a:
        return "n/a"
    return f"{(b - a) / a * 100:+.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--timeout", type=int, default=600, help="seconds per run")
    ap.add_argument("--arms", default="A,B")
    ap.add_argument(
        "--no-nudge",
        action="store_true",
        help="don't append the 'prefer search_code' system prompt in arm B",
    )
    ap.add_argument("--out", default=None, help="write raw rows + summary JSON here")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]

    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "mcp.json"
        cfg_path.write_text(json.dumps(mcp_config(repo)), encoding="utf-8")
        rows: list[dict] = []
        for task in tasks:
            for rep in range(args.repeats):
                for arm in arms:
                    cmd = build_cmd(
                        task["prompt"],
                        arm,
                        repo,
                        cfg_path,
                        args.model,
                        args.max_turns,
                        not args.no_nudge,
                    )
                    if args.dry_run:
                        print(" ".join(repr(c) if " " in c else c for c in cmd))
                        continue
                    print(f"[{task['id']}] rep {rep + 1} arm {arm} …", end=" ", flush=True)
                    r = run_once(cmd, repo, args.timeout)
                    r.update({"task": task["id"], "arm": arm, "rep": rep + 1})
                    r["pass"] = (
                        passed(r.get("result", ""), task.get("expect", []))
                        if "error" not in r
                        else None
                    )
                    rows.append(r)
                    print(
                        r.get("error")
                        or f"{r['tokens_total']} tok, {r['turns']} turns, ${r['cost_usd']:.4f}, pass={r['pass']}"
                    )
        if args.dry_run:
            return 0

    summary = {arm: summarize([r for r in rows if r["arm"] == arm]) for arm in arms}
    print("\n== Summary ==")
    for arm in arms:
        print(arm, json.dumps(summary[arm]))
    if "A" in summary and "B" in summary and summary["A"].get("n") and summary["B"].get("n"):
        a, b = summary["A"], summary["B"]
        print(
            f"\nΔ tokens (mean): {pct(a['tokens_total_mean'], b['tokens_total_mean'])}   "
            f"Δ cost: {pct(a['cost_usd_mean'], b['cost_usd_mean'])}   "
            f"Δ turns: {pct(a['turns_mean'], b['turns_mean'])}   "
            f"pass A/B: {a['pass_rate']} / {b['pass_rate']}"
        )
    if args.out:
        Path(args.out).write_text(
            json.dumps({"repo": str(repo), "summary": summary, "rows": rows}, indent=1),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
