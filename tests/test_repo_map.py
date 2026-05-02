from __future__ import annotations

from pathlib import Path

from token_reducer.db import connect_db, upsert_document
from token_reducer.repo_map import build_repo_map


def test_repo_map_marks_tests_and_entry_points(tmp_path: Path) -> None:
    test_py = tmp_path / "tests" / "test_api.py"
    test_py.parent.mkdir(parents=True)
    test_py.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    main_py = tmp_path / "main.py"
    main_py.write_text("if __name__ == '__main__':\n    pass\n", encoding="utf-8")

    conn = connect_db(tmp_path / "map.db")
    for p in (test_py, main_py):
        text = p.read_text(encoding="utf-8")
        upsert_document(
            conn=conn,
            source=str(p),
            raw_text=text,
            cleaned_text=text,
            chunk_size_words=80,
            overlap_words=0,
            dimensions=64,
            embedding_backend="hash",
            embedding_model=None,
        )
    conn.commit()

    rm = build_repo_map(conn)
    conn.close()

    assert str(test_py) in rm.test_sources
    assert str(main_py) in rm.entry_sources


def test_top_sources_for_task_navigate_orders_entry(tmp_path: Path) -> None:
    svc = tmp_path / "api" / "routes.py"
    svc.parent.mkdir(parents=True)
    svc.write_text("# routes\n", encoding="utf-8")
    main_py = tmp_path / "main.py"
    main_py.write_text("def main():\n    pass\n", encoding="utf-8")
    conn = connect_db(tmp_path / "nav.db")
    for p in (svc, main_py):
        t = p.read_text(encoding="utf-8")
        upsert_document(
            conn=conn,
            source=str(p),
            raw_text=t,
            cleaned_text=t,
            chunk_size_words=80,
            overlap_words=0,
            dimensions=64,
            embedding_backend="hash",
            embedding_model=None,
        )
    conn.commit()
    rm = build_repo_map(conn)
    conn.close()
    top = rm.top_sources_for_task("navigate", limit=5)
    assert str(main_py) in top
    assert top[0] == str(main_py) or str(main_py) in top[:2]
