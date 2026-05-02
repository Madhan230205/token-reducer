from __future__ import annotations

from pathlib import Path

from token_reducer.context_expansion import expand_context_candidates
from token_reducer.db import connect_db, upsert_document
from token_reducer.repo_map import RepoFileRecord, RepoMap
from token_reducer.retriever import fts_retrieve


def test_expand_context_adds_adjacent_chunks(tmp_path: Path) -> None:
    body = "def big():\n" + "".join([f"    x_{i} = {i}\n" for i in range(80)])
    src = tmp_path / "many.py"
    src.write_text(body, encoding="utf-8")
    conn = connect_db(tmp_path / "exp.db")
    t = src.read_text()
    upsert_document(
        conn,
        str(src),
        raw_text=t,
        cleaned_text=t,
        chunk_size_words=8,
        overlap_words=0,
        dimensions=64,
        embedding_backend="hash",
        embedding_model=None,
    )
    conn.commit()
    n_chunks = int(conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])
    assert n_chunks >= 2
    hits = fts_retrieve(conn, "x_40", limit=4)
    assert hits
    rm = RepoMap(
        files=(RepoFileRecord(str(src), "unknown", False, False),),
        test_sources=frozenset(),
        entry_sources=frozenset(),
        config_sources=frozenset(),
        service_sources=frozenset(),
        utility_sources=frozenset(),
        helper_sources=frozenset(),
        symbol_name_hints=frozenset(),
    )
    extra = expand_context_candidates(
        conn,
        hits,
        rm,
        frozenset(),
        include_callers=True,
        include_callees=False,
        max_extra=20,
    )
    hit_ids = {h.chunk_id for h in hits}
    new_ids = {c.chunk_id for c in extra if c.chunk_id not in hit_ids}
    assert len(new_ids) >= 1
    conn.close()


def test_expand_meta_token_pulls_matching_chunk(tmp_path: Path) -> None:
    conn = connect_db(tmp_path / "meta.db")
    p1 = tmp_path / "a.py"
    p1.write_text("def UniqueSymbolX():\n    return 1\n", encoding="utf-8")
    t = p1.read_text()
    upsert_document(
        conn,
        str(p1),
        raw_text=t,
        cleaned_text=t,
        chunk_size_words=40,
        overlap_words=0,
        dimensions=64,
        embedding_backend="hash",
        embedding_model=None,
    )
    conn.commit()
    hits = fts_retrieve(conn, "UniqueSymbolX", limit=2)
    assert hits
    rm = RepoMap(
        files=(RepoFileRecord(str(p1), "unknown", False, False),),
        test_sources=frozenset(),
        entry_sources=frozenset(),
        config_sources=frozenset(),
        service_sources=frozenset(),
        utility_sources=frozenset(),
        helper_sources=frozenset(),
        symbol_name_hints=frozenset({"UniqueSymbolX"}),
    )
    extra = expand_context_candidates(
        conn,
        hits,
        rm,
        frozenset({"UniqueSymbolX"}),
        include_callers=False,
        include_callees=True,
        max_extra=15,
    )
    combined = hits + extra
    assert any("UniqueSymbolX" in c.text for c in combined)
    conn.close()
