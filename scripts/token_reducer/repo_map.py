"""Lightweight repo map over the SQLite index — roles and entry points before raw chunk flood."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoFileRecord:
    """One indexed file with coarse role hints (path heuristics only)."""

    source: str
    role: str  # test | config | entry | service | utility | helper | unknown
    is_test: bool
    is_entry_point: bool


@dataclass(frozen=True)
class RepoMap:
    """Fast lookup over indexed documents — O(files) + small optional meta sample."""

    files: tuple[RepoFileRecord, ...]
    test_sources: frozenset[str]
    entry_sources: frozenset[str]
    config_sources: frozenset[str]
    service_sources: frozenset[str]
    utility_sources: frozenset[str]
    helper_sources: frozenset[str]
    symbol_name_hints: frozenset[str]

    def record_for(self, source: str) -> RepoFileRecord | None:
        for r in self.files:
            if r.source == source:
                return r
        return None

    def likely_test_sources(self) -> frozenset[str]:
        return self.test_sources

    def likely_entry_sources(self) -> frozenset[str]:
        return self.entry_sources

    def likely_config_sources(self) -> frozenset[str]:
        return self.config_sources

    def likely_service_sources(self) -> frozenset[str]:
        return self.service_sources

    def likely_utility_sources(self) -> frozenset[str]:
        return self.utility_sources | self.helper_sources

    def top_sources_for_task(
        self,
        task_mode: str,
        *,
        limit: int = 40,
        include_tests: bool = True,
        include_entry: bool = True,
        include_config: bool = False,
        include_callers: bool = False,
        include_callees: bool = False,
    ) -> tuple[str, ...]:
        """Ranked file paths for a task — score each file with simple role weights."""
        scored: list[tuple[int, str]] = []
        for rec in self.files:
            w = 0
            if include_tests and rec.is_test:
                w += _pri_test(task_mode)
            if include_entry and rec.is_entry_point:
                w += _pri_entry(task_mode)
            if include_config and rec.role == "config":
                w += _pri_config(task_mode)
            if include_callers and rec.role == "service":
                w += _pri_service(task_mode)
            if include_callees and rec.role in ("utility", "helper"):
                w += _pri_utility(task_mode)
            if task_mode == "navigate":
                if rec.is_entry_point:
                    w += 5
                elif rec.role == "service":
                    w += 2
                elif rec.role not in ("test", "config"):
                    w += 1
            if w > 0:
                scored.append((w, rec.source))
        scored.sort(key=lambda x: (-x[0], x[1]))
        out = [src for _w, src in scored[:limit]]
        return tuple(out)

    def boosted_sources_for_task(
        self,
        task_mode: str,
        query_tokens: frozenset[str],
        *,
        include_tests: bool,
        include_entry_points: bool,
        include_callers: bool,
        include_callees: bool,
        cap: int = 56,
    ) -> frozenset[str]:
        """Union of role-based priorities + paths whose stem matches query identifiers."""
        acc: list[str] = []
        acc.extend(self.top_sources_for_task(
            task_mode,
            limit=cap // 2,
            include_tests=include_tests,
            include_entry=include_entry_points,
            include_config=task_mode in ("debug", "add_feature"),
            include_callers=include_callers,
            include_callees=include_callees,
        ))
        # Symbol hints from index sample: boost files if hint appears in query
        qt = {t.lower() for t in query_tokens}
        for hint in self.symbol_name_hints:
            if hint.lower() in qt:
                for rec in self.files:
                    if hint.lower() in Path(rec.source).stem.lower():
                        acc.append(rec.source)
        # Filename token overlap (still O(files))
        for rec in self.files:
            stem = Path(rec.source).stem.lower()
            if any(tok.lower() in stem for tok in query_tokens if len(tok) > 2):
                acc.append(rec.source)
        seen: set[str] = set()
        uniq: list[str] = []
        for s in acc:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
            if len(uniq) >= cap:
                break
        return frozenset(uniq)


def _pri_test(mode: str) -> int:
    return 5 if mode in ("debug", "write_test", "refactor") else 1


def _pri_entry(mode: str) -> int:
    return 4 if mode in ("add_feature", "explain", "navigate", "debug") else 2


def _pri_config(mode: str) -> int:
    return 4 if mode in ("debug", "add_feature") else 0


def _pri_service(mode: str) -> int:
    return 4 if mode in ("debug", "refactor", "add_feature") else 1


def _pri_utility(mode: str) -> int:
    return 3 if mode in ("debug", "add_feature", "write_test") else 1


_TEST_PATH_RE = re.compile(
    r"(^|[\\/])(tests?|__tests__|testing|spec|_test\.py|\.test\.|\.spec\.)",
    re.I,
)
_TEST_NAME_RE = re.compile(r"^test_.*\.py$|.*_test\.py$|.*\.test\.ts$|.*\.spec\.[tj]s$", re.I)


def _classify_file(source: str) -> RepoFileRecord:
    p = source.replace("\\", "/")
    low = p.lower()
    name = Path(p).name.lower()

    is_test = bool(_TEST_PATH_RE.search(p) or _TEST_NAME_RE.match(name))
    is_entry = name in {
        "main.py",
        "__main__.py",
        "cli.py",
        "app.py",
        "manage.py",
        "run.py",
        "index.ts",
        "index.js",
        "main.go",
        "main.rs",
    }
    is_config = any(
        x in low
        for x in (
            ".env",
            "docker-compose",
            "pyproject.toml",
            "package.json",
            "tsconfig",
            "config.",
            "settings.",
            "application.",
        )
    )

    if is_test:
        role = "test"
    elif is_config:
        role = "config"
    elif is_entry:
        role = "entry"
    elif any(x in low for x in ("/api/", "/routes/", "/handlers/", "/controllers/", "/services/")):
        role = "service"
    elif "/utils/" in low or "/helpers/" in low or "util" in name:
        role = "utility"
    elif "/helper" in low or name.startswith("helper") or "helpers.py" in name:
        role = "helper"
    else:
        role = "unknown"

    return RepoFileRecord(
        source=source,
        role=role,
        is_test=is_test,
        is_entry_point=is_entry,
    )


def _sample_symbol_hints(conn: sqlite3.Connection, *, row_limit: int = 500) -> frozenset[str]:
    """Distinct symbol_name values from chunk meta_json sample — O(rows), not full table scan."""
    hints: set[str] = set()
    try:
        cur = conn.execute(
            """
            SELECT meta_json FROM chunks
            WHERE meta_json IS NOT NULL AND meta_json != ''
            LIMIT ?
            """,
            (row_limit,),
        )
        for row in cur.fetchall():
            raw = row[0]
            if not raw or not isinstance(raw, str):
                continue
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            sym = data.get("symbol_name") if isinstance(data, dict) else None
            if isinstance(sym, str) and 1 < len(sym) < 120:
                hints.add(sym)
    except sqlite3.Error:
        pass
    return frozenset(hints)


def build_repo_map(conn: sqlite3.Connection, *, limit: int = 8000) -> RepoMap:
    """Scan indexed document paths — O(files); optional small chunk meta sample for symbols."""
    rows = conn.execute(
        "SELECT source FROM documents ORDER BY source LIMIT ?",
        (limit,),
    ).fetchall()
    files: list[RepoFileRecord] = []
    tests: set[str] = set()
    entries: set[str] = set()
    configs: set[str] = set()
    services: set[str] = set()
    utils: set[str] = set()
    helpers: set[str] = set()
    for row in rows:
        src = str(row["source"])
        rec = _classify_file(src)
        files.append(rec)
        if rec.is_test:
            tests.add(src)
        if rec.is_entry_point:
            entries.add(src)
        if rec.role == "config":
            configs.add(src)
        if rec.role == "service":
            services.add(src)
        if rec.role == "utility":
            utils.add(src)
        if rec.role == "helper":
            helpers.add(src)
    sym_hints = _sample_symbol_hints(conn)
    return RepoMap(
        files=tuple(files),
        test_sources=frozenset(tests),
        entry_sources=frozenset(entries),
        config_sources=frozenset(configs),
        service_sources=frozenset(services),
        utility_sources=frozenset(utils),
        helper_sources=frozenset(helpers),
        symbol_name_hints=sym_hints,
    )
