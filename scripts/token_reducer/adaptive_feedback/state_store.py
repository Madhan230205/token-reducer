"""Durable staging / committed snapshots with atomic promote (spec Section 7)."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .models import CommittedActuators, StagingState


def adaptive_dir(workspace_root: Path | None) -> Path:
    """Per-workspace ``.token-reducer/adaptive`` when root known; else global plugin/home cache."""
    if workspace_root is not None:
        return workspace_root / ".token-reducer" / "adaptive"
    plug = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if plug:
        return Path(plug) / ".cache" / "token-reducer" / "adaptive"
    return Path.home() / ".cache" / "token-reducer" / "adaptive"


def staging_path(workspace_root: Path | None) -> Path:
    return adaptive_dir(workspace_root) / "staging.json"


def committed_path(workspace_root: Path | None) -> Path:
    return adaptive_dir(workspace_root) / "committed.json"


def committed_backup_path(workspace_root: Path | None) -> Path:
    return adaptive_dir(workspace_root) / "committed.json.bak"


def _staging_to_json(staging: StagingState) -> dict[str, Any]:
    """Serialize cohort dicts with JSON-object keys (stringified cohort tuples)."""
    return {
        "cohort_utility": {
            json.dumps(list(k), separators=(",", ":")): v for k, v in staging.cohort_utility.items()
        },
        "cohort_penalty": {
            json.dumps(list(k), separators=(",", ":")): v for k, v in staging.cohort_penalty.items()
        },
        "samples_per_cohort": {
            json.dumps(list(k), separators=(",", ":")): int(v) for k, v in staging.samples_per_cohort.items()
        },
    }


def _json_to_staging(data: dict[str, Any]) -> StagingState | None:
    try:
        cu_raw = data["cohort_utility"]
        cp_raw = data["cohort_penalty"]
        sp_raw = data["samples_per_cohort"]
        if not all(isinstance(x, dict) for x in (cu_raw, cp_raw, sp_raw)):
            return None

        def parse_floats(raw: dict[str, Any]) -> dict[tuple[str, ...], float]:
            out: dict[tuple[str, ...], float] = {}
            for k_str, v in raw.items():
                try:
                    key_list = json.loads(k_str)
                except json.JSONDecodeError:
                    continue
                if not isinstance(key_list, list):
                    continue
                tup = tuple(str(x) for x in key_list)
                out[tup] = float(v)
            return out

        def parse_samples(raw: dict[str, Any]) -> dict[tuple[str, ...], int]:
            out: dict[tuple[str, ...], int] = {}
            for k_str, v in raw.items():
                try:
                    key_list = json.loads(k_str)
                except json.JSONDecodeError:
                    continue
                if not isinstance(key_list, list):
                    continue
                tup = tuple(str(x) for x in key_list)
                out[tup] = int(v)
            return out

        return StagingState(
            cohort_utility=parse_floats(cu_raw),
            cohort_penalty=parse_floats(cp_raw),
            samples_per_cohort=parse_samples(sp_raw),
        )
    except (KeyError, TypeError, ValueError):
        return None


def save_staging(workspace_root: Path | None, staging: StagingState) -> None:
    path = staging_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_staging_to_json(staging), indent=2)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def load_staging(workspace_root: Path | None) -> StagingState:
    path = staging_path(workspace_root)
    if not path.is_file():
        return StagingState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return StagingState()
    if not isinstance(data, dict):
        return StagingState()
    st = _json_to_staging(data)
    return st if st is not None else StagingState()


def committed_to_json(c: CommittedActuators) -> dict[str, Any]:
    return {
        "retrieval_scale_mult_delta": c.retrieval_scale_mult_delta,
        "relevance_floor_delta": c.relevance_floor_delta,
        "prune_bias_ema_delta": dict(c.prune_bias_ema_delta),
        "skill_prior_delta": dict(c.skill_prior_delta),
    }


def json_to_committed(data: dict[str, Any]) -> CommittedActuators | None:
    try:
        pb = data.get("prune_bias_ema_delta")
        sk = data.get("skill_prior_delta")
        if not isinstance(pb, dict) or not isinstance(sk, dict):
            return None
        return CommittedActuators(
            retrieval_scale_mult_delta=float(data["retrieval_scale_mult_delta"]),
            relevance_floor_delta=float(data["relevance_floor_delta"]),
            prune_bias_ema_delta={str(k): float(v) for k, v in pb.items()},
            skill_prior_delta={str(k): float(v) for k, v in sk.items()},
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_committed(workspace_root: Path | None) -> CommittedActuators | None:
    path = committed_path(workspace_root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return json_to_committed(data)


def promote_committed_atomic(workspace_root: Path | None, committed: CommittedActuators) -> None:
    """Write ``committed.json`` via temp + rename; preserve previous as ``committed.json.bak``."""
    path = committed_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    bak = committed_backup_path(workspace_root)
    if path.is_file():
        with contextlib.suppress(OSError):
            shutil.copy2(path, bak)
    payload = json.dumps(committed_to_json(committed), indent=2)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def restore_committed_from_backup(workspace_root: Path | None) -> bool:
    """If ``committed.json`` is missing or corrupt, restore from ``committed.json.bak``."""
    path = committed_path(workspace_root)
    bak = committed_backup_path(workspace_root)
    if not bak.is_file():
        return False
    try:
        shutil.copy2(bak, path)
    except OSError:
        return False
    return True
