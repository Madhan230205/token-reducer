#!/usr/bin/env python3
"""Apply diff patches from Claude's SEARCH/REPLACE or AST-targeted format.

Protocols supported
-------------------
1. String SEARCH/REPLACE (original, brittle but universal):

   <<<< SEARCH [path/to/file.py]
   exact code to find
   ==== REPLACE
   new code to insert
   >>>>

2. AST-targeted replacement (new, requires tree-sitter optional deps):

   <<<< REPLACE_FUNCTION functionName [path/to/file.py]
   ...full new function body...
   >>>>

   <<<< REPLACE_CLASS ClassName [path/to/file.py]
   ...full new class body...
   >>>>

   <<<< REPLACE_METHOD ClassName.methodName [path/to/file.py]
   ...full new method body...
   >>>>

   The AST variants locate the node by name using tree-sitter, so
   minor whitespace/indentation differences never cause a miss-match.
   Falls back to string-replace automatically if tree-sitter is absent.

Transactional safety
--------------------
All blocks targeting the same file are staged in-memory first.
The file is only written to disk if every block for that file succeeds.
On any block failure the entire file transaction is rolled back (original
content preserved on disk).

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
from typing import Literal

# ---------------------------------------------------------------------------
# Tree-sitter optional import
# ---------------------------------------------------------------------------

try:
    import tree_sitter_c as _tsc
    import tree_sitter_cpp as _tscpp
    import tree_sitter_go as _tsgo
    import tree_sitter_java as _tsjava
    import tree_sitter_javascript as _tsjs
    import tree_sitter_python as _tspy
    import tree_sitter_ruby as _tsruby
    import tree_sitter_rust as _tsrust
    import tree_sitter_typescript as _tsts
    from tree_sitter import Language
    from tree_sitter import Parser as TSParser

    _TS_LANGUAGES: dict[str, Language] = {
        "py": Language(_tspy.language()),
        "js": Language(_tsjs.language()),
        "jsx": Language(_tsjs.language()),
        "ts": Language(_tsts.language_typescript()),
        "tsx": Language(_tsts.language_tsx()),
        "java": Language(_tsjava.language()),
        "go": Language(_tsgo.language()),
        "rs": Language(_tsrust.language()),
        "c": Language(_tsc.language()),
        "h": Language(_tsc.language()),
        "cpp": Language(_tscpp.language()),
        "cc": Language(_tscpp.language()),
        "rb": Language(_tsruby.language()),
    }
    _TS_AVAILABLE = True
except Exception:  # noqa: BLE001
    _TS_AVAILABLE = False
    _TS_LANGUAGES = {}

# ---------------------------------------------------------------------------
# Node-type tables
# ---------------------------------------------------------------------------

_NODE_TYPES: dict[tuple[str, str], list[str]] = {
    ("py", "function"): ["function_definition"],
    ("py", "class"): ["class_definition"],
    ("py", "method"): ["function_definition"],
    ("js", "function"): ["function_declaration", "function_expression", "arrow_function"],
    ("js", "class"): ["class_declaration"],
    ("js", "method"): ["method_definition"],
    ("jsx", "function"): ["function_declaration", "function_expression", "arrow_function"],
    ("jsx", "class"): ["class_declaration"],
    ("jsx", "method"): ["method_definition"],
    ("ts", "function"): ["function_declaration", "arrow_function"],
    ("ts", "class"): ["class_declaration"],
    ("ts", "method"): ["method_definition", "public_field_definition"],
    ("tsx", "function"): ["function_declaration", "arrow_function"],
    ("tsx", "class"): ["class_declaration"],
    ("tsx", "method"): ["method_definition"],
    ("java", "function"): ["method_declaration"],
    ("java", "class"): ["class_declaration"],
    ("java", "method"): ["method_declaration"],
    ("go", "function"): ["function_declaration"],
    ("go", "class"): ["type_declaration"],
    ("go", "method"): ["method_declaration"],
    ("rs", "function"): ["function_item"],
    ("rs", "class"): ["struct_item", "impl_item"],
    ("rs", "method"): ["function_item"],
    ("c", "function"): ["function_definition"],
    ("c", "class"): ["struct_specifier"],
    ("c", "method"): ["function_definition"],
    ("h", "function"): ["function_definition"],
    ("h", "class"): ["struct_specifier"],
    ("cpp", "function"): ["function_definition"],
    ("cpp", "class"): ["class_specifier"],
    ("cpp", "method"): ["function_definition"],
    ("cc", "function"): ["function_definition"],
    ("cc", "class"): ["class_specifier"],
    ("cc", "method"): ["function_definition"],
    ("rb", "function"): ["method"],
    ("rb", "class"): ["class"],
    ("rb", "method"): ["method", "singleton_method"],
}


def _get_node_types(ext: str, kind: str) -> list[str]:
    return _NODE_TYPES.get((ext, kind), _NODE_TYPES.get(("py", kind), []))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

BlockKind = Literal["search_replace", "replace_function", "replace_class", "replace_method"]


@dataclass
class DiffBlock:
    kind: BlockKind
    file_path: str | None
    search: str = ""
    replace: str = ""
    target_name: str = ""
    new_code: str = ""
    line_number: int = 0


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_SEARCH_REPLACE_RE = re.compile(
    r"<<<<\s*SEARCH(?:\s+([^\n]+))?\n"
    r"(.*?)\n"
    r"====\s*REPLACE\n"
    r"(.*?)\n"
    r">>>>",
    re.DOTALL,
)

_AST_BLOCK_RE = re.compile(
    r"<<<<\s*(REPLACE_FUNCTION|REPLACE_CLASS|REPLACE_METHOD)\s+"
    r"([\w.]+)"
    r"(?:\s+([^\n]+))?"
    r"\n(.*?)\n"
    r">>>>",
    re.DOTALL,
)


def parse_diff_blocks(content: str) -> list[DiffBlock]:
    blocks: list[DiffBlock] = []

    for match in _SEARCH_REPLACE_RE.finditer(content):
        file_path = match.group(1).strip() if match.group(1) else None
        blocks.append(
            DiffBlock(
                kind="search_replace",
                file_path=file_path,
                search=match.group(2),
                replace=match.group(3),
                line_number=content[: match.start()].count("\n") + 1,
            )
        )

    for match in _AST_BLOCK_RE.finditer(content):
        raw_kind = match.group(1).lower()
        kind: BlockKind = raw_kind  # type: ignore[assignment]
        target_name = match.group(2).strip()
        file_path = match.group(3).strip() if match.group(3) else None
        new_code = match.group(4)
        blocks.append(
            DiffBlock(
                kind=kind,
                file_path=file_path,
                target_name=target_name,
                new_code=new_code,
                line_number=content[: match.start()].count("\n") + 1,
            )
        )

    blocks.sort(key=lambda b: b.line_number)
    return blocks


# ---------------------------------------------------------------------------
# Content-based patch functions (operate on strings, no file I/O)
# Return (success, message, new_content)
# ---------------------------------------------------------------------------


def _patch_content_search_replace(block: DiffBlock, content: str) -> tuple[bool, str, str]:
    if block.search not in content:
        return False, "Search text not found", content

    occurrences = content.count(block.search)
    if occurrences > 1:
        return False, f"Search text found {occurrences} times (must be unique)", content

    return True, "ok", content.replace(block.search, block.replace, 1)


def _node_name(node) -> str | None:  # type: ignore[no-untyped-def]
    for child in node.children:
        if child.type in ("name", "identifier", "field_identifier", "property_identifier"):
            return child.text.decode()
    return None


def _find_named_nodes(root, node_types: list[str], name: str) -> list:
    results = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in node_types and _node_name(node) == name:
            results.append(node)
        stack.extend(reversed(node.children))
    return results


def _find_method_in_class(root, class_name: str, method_name: str, ext: str):
    class_types = _get_node_types(ext, "class")
    method_types = _get_node_types(ext, "method")
    for cls_node in _find_named_nodes(root, class_types, class_name):
        for method_node in _find_named_nodes(cls_node, method_types, method_name):
            return method_node
    return None


def _patch_content_ast(block: DiffBlock, content: str, ext: str) -> tuple[bool, str, str]:
    if not _TS_AVAILABLE:
        return _patch_content_ast_fallback(block, content)

    lang = _TS_LANGUAGES.get(ext)
    if lang is None:
        return _patch_content_ast_fallback(block, content)

    parser = TSParser(lang)
    src_bytes = content.encode("utf-8")
    tree = parser.parse(src_bytes)

    node = None
    if block.kind == "replace_method" and "." in block.target_name:
        class_name, _, method_name = block.target_name.partition(".")
        node = _find_method_in_class(tree.root_node, class_name, method_name, ext)
        if node is None:
            return False, f"Method '{method_name}' not found in class '{class_name}'", content
    else:
        kind_key = {
            "replace_function": "function",
            "replace_class": "class",
            "replace_method": "method",
        }[block.kind]
        node_types = _get_node_types(ext, kind_key)
        matches = _find_named_nodes(tree.root_node, node_types, block.target_name)
        if not matches:
            return False, f"'{block.target_name}' not found", content
        if len(matches) > 1:
            return (
                False,
                f"'{block.target_name}' found {len(matches)} times (must be unique)",
                content,
            )
        node = matches[0]

    start_byte = node.start_byte
    end_byte = node.end_byte

    # Detect original indentation
    line_start = src_bytes.rfind(b"\n", 0, start_byte) + 1
    indent = b""
    i = line_start
    while i < start_byte and src_bytes[i : i + 1] in (b" ", b"\t"):
        indent += src_bytes[i : i + 1]
        i += 1

    # Re-indent replacement to match original
    new_lines = block.new_code.rstrip("\n").split("\n")
    if new_lines:
        first_nonblank = next((ln for ln in new_lines if ln.strip()), "")
        strip_prefix = len(first_nonblank) - len(first_nonblank.lstrip())
        reindented = []
        for ln in new_lines:
            stripped = ln[strip_prefix:] if len(ln) >= strip_prefix else ln.lstrip()
            reindented.append(indent.decode() + stripped if stripped else stripped)
        new_code_bytes = "\n".join(reindented).encode("utf-8")
    else:
        new_code_bytes = block.new_code.encode("utf-8")

    new_src_bytes = src_bytes[:start_byte] + new_code_bytes + src_bytes[end_byte:]
    kind_label = block.kind.replace("replace_", "")
    return True, f"AST-replaced {kind_label} '{block.target_name}'", new_src_bytes.decode("utf-8")


def _patch_content_ast_fallback(block: DiffBlock, content: str) -> tuple[bool, str, str]:
    """Heuristic fallback when tree-sitter is unavailable."""
    name = block.target_name.split(".")[-1]
    patterns = [
        rf"^\s*(async\s+)?def\s+{re.escape(name)}\s*[:(]",
        rf"^\s*(export\s+)?(async\s+)?function\s+{re.escape(name)}\s*[(<]",
        rf"^\s*(public|private|protected|static|\s)*(async\s+)?\w+\s+{re.escape(name)}\s*\(",
        rf"^\s*func\s+\(.*?\)\s*{re.escape(name)}\s*\(",
        rf"^\s*func\s+{re.escape(name)}\s*\(",
        rf"^\s*(pub\s+)?fn\s+{re.escape(name)}\s*[(<]",
        rf"^\s*(class|struct)\s+{re.escape(name)}\s*[:({{]",
        rf"^\s*def\s+{re.escape(name)}\b",
    ]
    combined = re.compile("|".join(patterns), re.MULTILINE)
    m = combined.search(content)
    if not m:
        return False, f"Heuristic could not locate '{name}' (tree-sitter unavailable)", content

    lines = content.splitlines(keepends=True)
    def_line_idx = content[: m.start()].count("\n")
    def_indent = len(lines[def_line_idx]) - len(lines[def_line_idx].lstrip())

    end_line_idx = def_line_idx + 1
    while end_line_idx < len(lines):
        ln = lines[end_line_idx]
        stripped = ln.lstrip()
        if stripped and (len(ln) - len(stripped)) <= def_indent:
            break
        end_line_idx += 1

    old_block = "".join(lines[def_line_idx:end_line_idx])
    if old_block not in content:
        return False, f"Could not isolate block for '{name}'", content

    new_content = content.replace(old_block, block.new_code.rstrip("\n") + "\n", 1)
    return True, f"Heuristic-replaced '{name}' (tree-sitter unavailable)", new_content


def _patch_content(block: DiffBlock, content: str, ext: str) -> tuple[bool, str, str]:
    """Dispatch to the correct patcher and return (ok, msg, new_content)."""
    if block.kind == "search_replace":
        return _patch_content_search_replace(block, content)
    return _patch_content_ast(block, content, ext)


# ---------------------------------------------------------------------------
# Transactional apply_diffs
#
# All blocks for a given file are staged in-memory.  The file is written only
# if every block targeting it succeeds.  On any failure the original content
# is preserved on disk (transaction rollback).
# ---------------------------------------------------------------------------


def apply_diffs(
    content: str,
    working_dir: Path | None = None,
    default_file: Path | None = None,
    dry_run: bool = False,
) -> dict:
    blocks = parse_diff_blocks(content)
    if not blocks:
        return {"applied": 0, "failed": 0, "messages": ["No diff blocks found"]}

    working_dir = working_dir or Path.cwd()
    results: dict = {"applied": 0, "failed": 0, "messages": []}

    # Group blocks by resolved file path (preserving order within each file)
    file_groups: dict[Path, list[DiffBlock]] = {}
    orphans: list[DiffBlock] = []

    for block in blocks:
        if block.file_path:
            target = (working_dir / block.file_path).resolve()
        elif default_file:
            target = default_file.resolve()
        else:
            orphans.append(block)
            continue
        file_groups.setdefault(target, []).append(block)

    for block in orphans:
        results["failed"] += 1
        results["messages"].append(f"Block at line {block.line_number}: No file path specified")

    # Process each file as a transaction
    for target, file_blocks in file_groups.items():
        if not target.exists():
            for _ in file_blocks:
                results["failed"] += 1
                results["messages"].append(f"File not found: {target}")
            continue

        try:
            original = target.read_text(encoding="utf-8")
        except Exception as exc:
            for _ in file_blocks:
                results["failed"] += 1
            results["messages"].append(f"Failed to read {target}: {exc}")
            continue

        ext = target.suffix.lstrip(".")
        current = original
        tx_ok = True
        tx_messages: list[str] = []

        for block in file_blocks:
            ok, msg, current = _patch_content(block, current, ext)
            label = f"[{target.name}] {msg}"
            tx_messages.append(label)
            if not ok:
                tx_ok = False
                # Stop processing further blocks for this file
                remaining = len(file_blocks) - file_blocks.index(block) - 1
                if remaining:
                    tx_messages.append(
                        f"[{target.name}] Transaction rolled back — "
                        f"{remaining} subsequent block(s) skipped"
                    )
                break

        if tx_ok:
            if dry_run:
                results["applied"] += len(file_blocks)
                results["messages"].extend(f"[DRY RUN] {m}" for m in tx_messages)
            else:
                try:
                    target.write_text(current, encoding="utf-8")
                    results["applied"] += len(file_blocks)
                    results["messages"].extend(tx_messages)
                except Exception as exc:
                    results["failed"] += len(file_blocks)
                    results["messages"].append(f"Failed to write {target}: {exc}")
        else:
            # Entire file transaction failed — original untouched on disk
            results["failed"] += len(file_blocks)
            results["messages"].extend(tx_messages)
            results["messages"].append(
                f"[{target.name}] Original file preserved (transaction rolled back)"
            )

    return results


# ---------------------------------------------------------------------------
# Legacy single-block helper (used by tests / external callers)
# ---------------------------------------------------------------------------


def apply_block(block: DiffBlock, target_file: Path, dry_run: bool = False) -> tuple[bool, str]:
    """Apply a single block to a file.  Not transactional — prefer apply_diffs."""
    if not target_file.exists():
        return False, f"File not found: {target_file}"
    try:
        content = target_file.read_text(encoding="utf-8")
    except Exception as exc:
        return False, f"Failed to read {target_file}: {exc}"

    ext = target_file.suffix.lstrip(".")
    ok, msg, new_content = _patch_content(block, content, ext)
    if not ok:
        return False, f"{msg} in {target_file}"
    if dry_run:
        return True, f"[DRY RUN] Would apply to {target_file}: {msg}"
    try:
        target_file.write_text(new_content, encoding="utf-8")
    except Exception as exc:
        return False, f"Failed to write {target_file}: {exc}"
    return True, f"{msg} in {target_file}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply SEARCH/REPLACE or AST-targeted diff blocks from Claude's response"
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
        except Exception as exc:
            print(f"Error reading input: {exc}", file=sys.stderr)
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
        import json as _json

        print(_json.dumps(results, indent=2))
    else:
        for msg in results["messages"]:
            print(msg)
        print(f"\nApplied: {results['applied']}, Failed: {results['failed']}")

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
