"""Load ``apply_diff.apply_diffs`` from the repo ``scripts/apply_diff.py`` (no package install path)."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_apply_diffs_cache: Callable[..., dict[str, Any]] | None = None


def get_apply_diffs() -> Callable[..., dict[str, Any]]:
    global _apply_diffs_cache
    if _apply_diffs_cache is not None:
        return _apply_diffs_cache
    here = Path(__file__).resolve()
    scripts_dir = here.parent.parent
    mod_path = scripts_dir / "apply_diff.py"
    name = "_token_reducer_apply_diff_runtime"
    spec = importlib.util.spec_from_file_location(name, mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load apply_diff from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "apply_diffs", None)
    if fn is None:
        raise ImportError("apply_diffs not found in apply_diff.py")
    _apply_diffs_cache = fn
    return _apply_diffs_cache
