from __future__ import annotations

from token_reducer.intent import analyze_query_intent


def test_bug_fix_intent() -> None:
    assert analyze_query_intent("Fix crash when null pointer in auth handler") == "bug_fix"


def test_explain_intent() -> None:
    assert analyze_query_intent("What does this function do?") == "explain_code"


def test_navigation_intent() -> None:
    assert analyze_query_intent("Where is user_service.py") == "navigation"


def test_feature_intent() -> None:
    assert analyze_query_intent("Add new API endpoint for export") == "feature_add"
