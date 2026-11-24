from __future__ import annotations

import os

from token_reducer.execution_route import build_routing_plan, resolve_execution_route
from token_reducer.intent import detect_intent, structured_intent_to_dict


def test_output_cap_simple_below_complex() -> None:
    q = "thanks!"
    si = detect_intent(q)
    route = resolve_execution_route(q, si)
    plan = build_routing_plan(route, structured_intent_to_dict(si), workspace_root=None)
    q2 = "We need a migration plan for the entire auth stack step by step."
    si2 = detect_intent(q2)
    route2 = resolve_execution_route(q2, si2)
    plan2 = build_routing_plan(route2, structured_intent_to_dict(si2), workspace_root=None)
    assert plan.output_token_cap <= plan2.output_token_cap


def test_env_hard_cap() -> None:
    os.environ["TOKEN_REDUCER_OUTPUT_TOKEN_CAP"] = "900"
    try:
        q = "fix crash in handler.py traceback null"
        si = detect_intent(q)
        route = resolve_execution_route(q, si)
        plan = build_routing_plan(route, structured_intent_to_dict(si), workspace_root=None)
        assert plan.output_token_cap <= 900
    finally:
        del os.environ["TOKEN_REDUCER_OUTPUT_TOKEN_CAP"]
