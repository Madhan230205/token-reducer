#!/usr/bin/env python3
"""Apply diff patches from Claude's SEARCH/REPLACE format.

Usage:
    python apply_diff.py --input response.txt
    python apply_diff.py --input response.txt --dry-run
    echo "response" | python apply_diff.py --stdin
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DiffBlock:
    """Represents a single SEARCH/REPLACE block."""
    file_path: str | None
    search: str
    replace: str
    line_number: int


def parse_diff_blocks(content: str) -> list[DiffBlock]:
    """Parse SEARCH/REPLACE blocks from Claude's response.
    
    Expected format:
    <<<< SEARCH
    exact code to find
    ==== REPLACE
    new code to insert
    >>>>
    
    Or with file path:
    <<<< SEARCH path/to/file.py
    ...
    """
    blocks: list[DiffBlock] = []
    pattern = re.compile(
        r"<<<<\s*SEARCH(?:\s+([^\n]+))?\n"
        r"(.*?)\n"
        r"====\s*REPLACE\n"
        r"(.*?)\n"
        r">>>>",
        re.DOTALL
    )
    
    for match in pattern.finditer(content):
        file_path = match.group(1).strip() if match.group(1) else None
        blocks.append(DiffBlock(
            file_path=file_path,
            search=match.group(2),
            replace=match.group(3),
            line_number=content[:match.start()].count("\n") + 1,
        ))
    
    return blocks


def apply_block(block: DiffBlock, target_file: Path, dry_run: bool = False) -> tuple[bool, str]:
    """Apply a single diff block to a file."""
    if not target_file.exists():
        return False, f"File not found: {target_file}"
    
    try:
        content = target_file.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Failed to read {target_file}: {e}"
    
    if block.search not in content:
        return False, f"Search text not found in {target_file}"
    
    occurrences = content.count(block.search)
    if occurrences > 1:
        return False, f"Search text found {occurrences} times (must be unique)"
    
    new_content = content.replace(block.search, block.replace, 1)
    
    if dry_run:
        return True, f"[DRY RUN] Would apply patch to {target_file}"
    
    try:
        target_file.write_text(new_content, encoding="utf-8")
        return True, f"Applied patch to {target_file}"
    except Exception as e:
        return False, f"Failed to write {target_file}: {e}"


def apply_diffs(
    content: str,
    working_dir: Path | None = None,
    default_file: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Apply all diff blocks from content."""
    blocks = parse_diff_blocks(content)
    
    if not blocks:
        return {"applied": 0, "failed": 0, "messages": ["No SEARCH/REPLACE blocks found"]}
    
    working_dir = working_dir or Path.cwd()
    results = {"applied": 0, "failed": 0, "messages": []}
    
    for block in blocks:
        if block.file_path:
            target = working_dir / block.file_path
        elif default_file:
            target = default_file
        else:
            results["failed"] += 1
            results["messages"].append(f"Block at line {block.line_number}: No file path specified")
            continue
        
        success, message = apply_block(block, target, dry_run)
        results["applied" if success else "failed"] += 1
        results["messages"].append(message)
    
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply SEARCH/REPLACE diff blocks from Claude's response"
    )
    parser.add_argument("--input", "-i", help="Input file containing Claude's response")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    parser.add_argument("--file", "-f", help="Default file to patch if not specified in blocks")
    parser.add_argument("--dir", "-d", default=".", help="Working directory for resolving paths")
    parser.add_argument("--dry-run", action="store_true", help="Preview without modifying files")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    if args.stdin:
        content = sys.stdin.read()
    elif args.input:
        try:
            content = Path(args.input).read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading input: {e}", file=sys.stderr)
            return 1
    else:
        parser.print_help()
        return 1
    
    results = apply_diffs(
        content=content,
        working_dir=Path(args.dir),
        default_file=Path(args.file) if args.file else None,
        dry_run=args.dry_run,
    )
    
    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        for msg in results["messages"]:
            print(msg)
        print(f"\nApplied: {results['applied']}, Failed: {results['failed']}")
    
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())