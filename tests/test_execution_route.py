from __future__ import annotations

import os

from token_reducer.execution_route import (
    classify_execution_tier,
    resolve_execution_route,
    tier_override_from_env,
)
from token_reducer.intent import detect_intent


def test_simple_chat() -> None:
    si = detect_intent("thanks, that helped")
    assert classify_execution_tier("thanks, that helped", si) == "simple"


def test_tool_slash_command() -> None:
    si = detect_intent("/commit message please")
    assert classify_execution_tier("/commit message please", si) == "tool"


def test_complex_feature() -> None:
    q = (
        "We need to refactor the entire auth module and add OAuth2 support, "
        "then migrate all handlers — step by step plan please."
    )
    si = detect_intent(q)
    assert classify_execution_tier(q, si) == "complex"


def test_env_override() -> None:
    os.environ["TOKEN_REDUCER_TIER_OVERRIDE"] = "simple"
    try:
        assert tier_override_from_env() == "simple"
        si = detect_intent("some huge multi step thing that would be complex")
        assert classify_execution_tier(
            "implement everything across the repo with migrations", si
        ) == "simple"
    finally:
        os.environ.pop("TOKEN_REDUCER_TIER_OVERRIDE", None)


def test_resolve_route_skips_subagents_for_tool() -> None:
    os.environ.pop("TOKEN_REDUCER_TIER_OVERRIDE", None)
    q = "/skill run tests"
    r = resolve_execution_route(q, detect_intent(q))
    assert r.tier == "tool"
    assert r.skip_subagents is True
