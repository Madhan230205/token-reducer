"""Tests for actuators, guardrails, and state_store."""

from __future__ import annotations

from pathlib import Path

import pytest
from token_reducer.adaptive_feedback.actuators import committed_from_staging
from token_reducer.adaptive_feedback.guardrails import clamp_committed
from token_reducer.adaptive_feedback.models import CommittedActuators, StagingState
from token_reducer.adaptive_feedback.state_store import (
    committed_path,
    load_committed,
    load_staging,
    promote_committed_atomic,
    restore_committed_from_backup,
    save_staging,
)


def test_committed_neutral_when_under_sampled() -> None:
    st = StagingState()
    st.cohort_utility[("x",)] = 1.0
    st.samples_per_cohort[("x",)] = 3
    c = committed_from_staging(st, min_samples=8)
    assert c.retrieval_scale_mult_delta == 0.0


def test_committed_when_sampled() -> None:
    st = StagingState()
    st.cohort_utility[("x",)] = 0.5
    st.cohort_penalty[("x",)] = 0.0
    st.samples_per_cohort[("x",)] = 10
    c = committed_from_staging(st, min_samples=8)
    assert c.retrieval_scale_mult_delta != 0.0


def test_clamp_committed() -> None:
    wild = CommittedActuators(retrieval_scale_mult_delta=9.0, relevance_floor_delta=-9.0)
    c = clamp_committed(wild)
    assert abs(c.retrieval_scale_mult_delta) <= 0.06
    assert abs(c.relevance_floor_delta) <= 0.02


def test_staging_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    st = StagingState()
    st.cohort_utility[("tier", "sid")] = 0.3
    st.samples_per_cohort[("tier", "sid")] = 10
    save_staging(root, st)
    back = load_staging(root)
    assert back.cohort_utility[("tier", "sid")] == pytest.approx(0.3)
    assert back.samples_per_cohort[("tier", "sid")] == 10


def test_promote_and_restore_backup(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    promote_committed_atomic(
        root,
        CommittedActuators(retrieval_scale_mult_delta=0.01, relevance_floor_delta=0.005),
    )
    first = load_committed(root)
    assert first is not None
    assert first.retrieval_scale_mult_delta == pytest.approx(0.01)

    promote_committed_atomic(root, CommittedActuators(retrieval_scale_mult_delta=-0.02, relevance_floor_delta=0.0))
    second = load_committed(root)
    assert second is not None
    assert second.retrieval_scale_mult_delta == pytest.approx(-0.02)

    corrupt = committed_path(root)
    corrupt.write_text("{broken", encoding="utf-8")
    assert restore_committed_from_backup(root) is True
    restored = load_committed(root)
    assert restored is not None
    # .bak held previous snapshot (0.01) before last promote
    assert restored.retrieval_scale_mult_delta == pytest.approx(0.01)
