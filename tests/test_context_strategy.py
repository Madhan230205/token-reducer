from __future__ import annotations

from token_reducer.context_strategy import ContextStrategy, map_query_to_strategy


def _s(**kwargs: object) -> dict[str, object]:
    base: dict[str, object] = {
        "type": "code",
        "k": 40,
        "token_budget": 2000,
        "compression_level": "medium",
        "legacy_intent": "explain_code",
    }
    base.update(kwargs)
    return base


def test_chat_strategy_skips_heavy_stages() -> None:
    s = map_query_to_strategy("hello there", _s(type="chat", legacy_intent="explain_code"), "fts_only", use_vector=False)
    assert s.strategy_id == "conversational_light"
    assert s.skip_fusion is True
    assert s.skip_neighborhood is True
    assert s.prune_k <= 8
    assert "user" in s.attention_frame.lower()


def test_short_query_uses_light_shape() -> None:
    s = map_query_to_strategy("one two three four five six seven eight nine", _s(), "fts_only", use_vector=False)
    assert s.strategy_id == "conversational_light"


def test_bug_fix_hybrid_boosts_prune_when_vector() -> None:
    s = map_query_to_strategy(
        "the production service crashes with a full traceback on every deploy "
        "when the background worker starts",
        _s(legacy_intent="bug_fix"),
        "full_hybrid",
        use_vector=True,
    )
    assert s.strategy_id == "failure_adjacent"
    assert s.prune_k == 17
    assert s.skip_fusion is False


def test_bug_fix_no_vector_lower_prune() -> None:
    s = map_query_to_strategy(
        "error five hundred when calling the authenticated handler after the "
        "latest release in staging environment consistently",
        _s(legacy_intent="bug_fix"),
        "full_hybrid",
        use_vector=False,
    )
    assert s.prune_k == 13


def test_navigation_attention_frame() -> None:
    s = map_query_to_strategy(
        "where exactly in this repository is the function that validates JWT "
        "tokens for incoming requests",
        _s(legacy_intent="navigation"),
        "fts_only",
        use_vector=False,
    )
    assert s.strategy_id == "path_and_symbol"
    assert "path" in s.attention_frame.lower()


def test_to_dict_roundtrip_keys() -> None:
    s = ContextStrategy(
        strategy_id="x",
        merge_cap=30,
        prune_k=10,
        skip_fusion=True,
        skip_neighborhood=False,
        attention_frame="Focus.",
    )
    d = s.to_dict()
    assert d["strategy_id"] == "x"
    assert d["merge_cap"] == 30
    assert d["prune_k"] == 10
