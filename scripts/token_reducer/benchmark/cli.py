"""CLI for benchmark harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from .runner import run_suite

bench_app = typer.Typer(
    name="proof-harness",
    help="Proof harness: scenarios → JSON Lines artifacts (see docs/superpowers/specs/2026-05-02-benchmark-proof-harness-design.md).",
)


@bench_app.command("run")
def run_cmd(
    tier: Annotated[str, typer.Option(help="smoke | nightly | weekly")] = "smoke",
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root (default: current directory)"),
    ] = None,
    scenarios_dir: Annotated[
        Path | None,
        typer.Option(help="Override scenarios directory (default: <repo>/benchmarks/scenarios)"),
    ] = None,
    output_jsonl: Annotated[
        Path | None,
        typer.Option("--output-jsonl", help="Append JSON Lines results to this file"),
    ] = None,
    fail_fast: Annotated[
        bool, typer.Option("--fail-fast", help="Stop on first required failure")
    ] = False,
) -> None:
    """Run benchmark scenarios for the given tier."""
    root = (repo_root or Path.cwd()).resolve()
    rows, failed_required = run_suite(
        tier=tier,
        repo_root=root,
        scenarios_dir=scenarios_dir,
        fail_fast=fail_fast,
    )
    for row in rows:
        line = json.dumps(row, sort_keys=True, ensure_ascii=False)
        typer.echo(line)
        if output_jsonl is not None:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with output_jsonl.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    if failed_required:
        sys.exit(1)


def run_proof_harness_cli() -> None:
    bench_app()


if __name__ == "__main__":
    run_proof_harness_cli()
