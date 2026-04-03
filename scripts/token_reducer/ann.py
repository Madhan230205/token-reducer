from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .config import DEFAULT_ANN_EF_SEARCH, DEFAULT_ANN_ENGINE
from .db import get_index_fingerprint
from .models import ann_artifact_prefix, utc_now_iso


def try_import_hnsw_modules():
    try:
        import hnswlib  # type: ignore
        import numpy as np  # type: ignore

        return hnswlib, np
    except Exception:
        return None, None


def fetch_embeddings_for_ann(
    conn: sqlite3.Connection,
    backend: str,
    dimensions: int,
    model_name: str | None,
) -> list[sqlite3.Row]:
    sql = """
    SELECT chunk_id, embedding_json, dimensions, backend, model_name
    FROM chunk_embeddings
    WHERE backend = ? AND dimensions = ?
    """
    params: list[object] = [backend, dimensions]
    if backend == "ml" and model_name:
        sql += " AND (model_name = ? OR model_name IS NULL)"
        params.append(model_name)
    return conn.execute(sql, tuple(params)).fetchall()


def infer_embedding_dimensions(
    conn: sqlite3.Connection,
    backend: str,
    model_name: str | None,
    fallback_dimensions: int,
) -> int:
    sql = "SELECT dimensions FROM chunk_embeddings WHERE backend = ?"
    params: list[object] = [backend]
    if backend == "ml" and model_name:
        sql += " AND (model_name = ? OR model_name IS NULL)"
        params.append(model_name)
    sql += " ORDER BY rowid DESC LIMIT 1"
    row = conn.execute(sql, tuple(params)).fetchone()
    if row and row["dimensions"] is not None:
        return int(row["dimensions"])
    return fallback_dimensions


def build_hnsw_index(
    conn: sqlite3.Connection,
    db_path: Path,
    backend: str,
    dimensions: int,
    model_name: str | None,
) -> bool:
    hnswlib, np = try_import_hnsw_modules()
    if hnswlib is None or np is None:
        return False

    rows = fetch_embeddings_for_ann(
        conn=conn,
        backend=backend,
        dimensions=dimensions,
        model_name=model_name,
    )
    if not rows:
        return False

    ids: list[int] = []
    vectors: list[list[float]] = []
    for row in rows:
        try:
            vec = [float(x) for x in json.loads(str(row["embedding_json"]))]
        except Exception:
            continue
        if len(vec) != dimensions:
            continue
        ids.append(int(row["chunk_id"]))
        vectors.append(vec)

    if not ids:
        return False

    prefix = ann_artifact_prefix(
        db_path=db_path,
        backend=backend,
        dimensions=dimensions,
        model_name=model_name,
    )
    index_path = prefix.with_suffix(".hnsw.bin")
    meta_path = prefix.with_suffix(".hnsw.meta.json")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    idx = hnswlib.Index(space="cosine", dim=dimensions)
    idx.init_index(max_elements=len(ids), ef_construction=220, M=32)
    idx.add_items(np.array(vectors, dtype=np.float32), np.array(ids, dtype=np.int64))
    idx.set_ef(min(320, max(DEFAULT_ANN_EF_SEARCH, len(ids))))
    idx.save_index(str(index_path))

    meta_payload = {
        "engine": DEFAULT_ANN_ENGINE,
        "backend": backend,
        "dimensions": dimensions,
        "model_name": model_name,
        "elements": len(ids),
        "index_fingerprint": get_index_fingerprint(conn),
        "created_at": utc_now_iso(),
    }
    meta_path.write_text(json.dumps(meta_payload), encoding="utf-8")
    return True


def query_hnsw_index(
    conn: sqlite3.Connection,
    db_path: Path,
    query_vec: list[float],
    limit: int,
    backend: str,
    model_name: str | None,
) -> list[tuple[int, float]]:
    hnswlib, np = try_import_hnsw_modules()
    if hnswlib is None or np is None:
        return []

    dimensions = len(query_vec)
    prefix = ann_artifact_prefix(
        db_path=db_path,
        backend=backend,
        dimensions=dimensions,
        model_name=model_name,
    )
    index_path = prefix.with_suffix(".hnsw.bin")
    meta_path = prefix.with_suffix(".hnsw.meta.json")
    if not index_path.exists() or not meta_path.exists():
        built = build_hnsw_index(
            conn=conn,
            db_path=db_path,
            backend=backend,
            dimensions=dimensions,
            model_name=model_name,
        )
        if not built:
            return []

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    current_fp = get_index_fingerprint(conn)
    if meta.get("index_fingerprint") != current_fp:
        rebuilt = build_hnsw_index(
            conn=conn,
            db_path=db_path,
            backend=backend,
            dimensions=dimensions,
            model_name=model_name,
        )
        if not rebuilt:
            return []

    idx = hnswlib.Index(space="cosine", dim=dimensions)
    idx.load_index(str(index_path))
    idx.set_ef(DEFAULT_ANN_EF_SEARCH)

    current_count = int(idx.get_current_count())
    if current_count <= 0:
        return []

    k = min(max(1, limit), current_count)
    labels, distances = idx.knn_query(np.array([query_vec], dtype=np.float32), k=k)
    results: list[tuple[int, float]] = []
    for label, distance in zip(labels[0], distances[0], strict=False):
        if int(label) < 0:
            continue
        sim = max(0.0, 1.0 - float(distance))
        results.append((int(label), sim))
    return results


def try_import_faiss():
    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore

        return faiss, np
    except Exception:
        return None, None


def build_faiss_index(
    conn: sqlite3.Connection,
    db_path: Path,
    backend: str,
    dimensions: int,
    model_name: str | None,
) -> bool:
    import math

    faiss, np = try_import_faiss()
    if faiss is None or np is None:
        return False

    rows = fetch_embeddings_for_ann(
        conn=conn,
        backend=backend,
        dimensions=dimensions,
        model_name=model_name,
    )
    if not rows:
        return False

    ids: list[int] = []
    vectors: list[list[float]] = []
    for row in rows:
        try:
            vec = [float(x) for x in json.loads(str(row["embedding_json"]))]
        except Exception:
            continue
        if len(vec) != dimensions:
            continue
        ids.append(int(row["chunk_id"]))
        vectors.append(vec)

    if not ids:
        return False

    prefix = ann_artifact_prefix(
        db_path=db_path,
        backend=backend,
        dimensions=dimensions,
        model_name=model_name,
    )
    index_path = prefix.with_suffix(".faiss.bin")
    meta_path = prefix.with_suffix(".faiss.meta.json")
    idmap_path = prefix.with_suffix(".faiss.idmap.json")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(data)

    n_elements = len(ids)
    if n_elements < 1000:
        index = faiss.IndexFlatIP(dimensions)
    else:
        n_clusters = min(int(math.sqrt(n_elements)), 256)
        quantizer = faiss.IndexFlatIP(dimensions)
        index = faiss.IndexIVFFlat(quantizer, dimensions, n_clusters, faiss.METRIC_INNER_PRODUCT)
        index.train(data)
        index.nprobe = min(16, n_clusters)

    index.add(data)
    faiss.write_index(index, str(index_path))

    idmap_path.write_text(json.dumps(ids, separators=(",", ":")), encoding="utf-8")
    meta_payload = {
        "engine": "faiss",
        "backend": backend,
        "dimensions": dimensions,
        "model_name": model_name,
        "elements": n_elements,
        "index_fingerprint": get_index_fingerprint(conn),
        "created_at": utc_now_iso(),
    }
    meta_path.write_text(json.dumps(meta_payload), encoding="utf-8")
    return True


def query_faiss_index(
    conn: sqlite3.Connection,
    db_path: Path,
    query_vec: list[float],
    limit: int,
    backend: str,
    model_name: str | None,
) -> list[tuple[int, float]]:
    faiss, np = try_import_faiss()
    if faiss is None or np is None:
        return []

    dimensions = len(query_vec)
    prefix = ann_artifact_prefix(
        db_path=db_path,
        backend=backend,
        dimensions=dimensions,
        model_name=model_name,
    )
    index_path = prefix.with_suffix(".faiss.bin")
    meta_path = prefix.with_suffix(".faiss.meta.json")
    idmap_path = prefix.with_suffix(".faiss.idmap.json")

    if not index_path.exists() or not meta_path.exists():
        built = build_faiss_index(
            conn=conn,
            db_path=db_path,
            backend=backend,
            dimensions=dimensions,
            model_name=model_name,
        )
        if not built:
            return []

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    current_fp = get_index_fingerprint(conn)
    if meta.get("index_fingerprint") != current_fp:
        rebuilt = build_faiss_index(
            conn=conn,
            db_path=db_path,
            backend=backend,
            dimensions=dimensions,
            model_name=model_name,
        )
        if not rebuilt:
            return []

    try:
        id_map = json.loads(idmap_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    index = faiss.read_index(str(index_path))
    qvec = np.array([query_vec], dtype=np.float32)
    faiss.normalize_L2(qvec)

    k = min(max(1, limit), index.ntotal)
    scores, indices = index.search(qvec, k)

    results: list[tuple[int, float]] = []
    for idx_pos, score in zip(indices[0], scores[0], strict=False):
        idx_pos = int(idx_pos)
        if idx_pos < 0 or idx_pos >= len(id_map):
            continue
        chunk_id = id_map[idx_pos]
        sim = max(0.0, float(score))
        results.append((chunk_id, sim))
    return results
