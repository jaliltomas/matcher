import hashlib
import logging
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def product_cache_key(product: dict[str, Any]) -> str:
    def _norm(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = " ".join(text.split())
        return text

    sitio = _norm(product.get("sitio"))
    seller = _norm(product.get("seller"))
    url = _norm(product.get("url_producto"))
    raw = f"{sitio}|{seller}|{url}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class ProductEmbeddingCache:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        return connection

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS product_embeddings (
                    model_sig TEXT NOT NULL,
                    product_key TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector_blob BLOB NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (model_sig, product_key)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_product_embeddings_model ON product_embeddings(model_sig)"
            )

    def get_many(self, model_sig: str, keys: list[str]) -> dict[str, np.ndarray]:
        if not keys:
            return {}
        placeholders = ",".join(["?"] * len(keys))
        query = (
            f"SELECT product_key, dim, vector_blob FROM product_embeddings "
            f"WHERE model_sig = ? AND product_key IN ({placeholders})"
        )
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, [model_sig, *keys]).fetchall()

        found: dict[str, np.ndarray] = {}
        for key, dim, blob in rows:
            vector = np.frombuffer(blob, dtype=np.float32)
            if int(dim) != int(vector.shape[0]):
                continue
            found[str(key)] = vector
        return found

    def put_many(self, model_sig: str, vectors_by_key: dict[str, np.ndarray]) -> None:
        if not vectors_by_key:
            return

        now = time.time()
        rows: list[tuple[str, str, int, bytes, float]] = []
        for key, vec in vectors_by_key.items():
            arr = np.asarray(vec, dtype=np.float32)
            rows.append((model_sig, key, int(arr.shape[0]), arr.tobytes(), now))

        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO product_embeddings(model_sig, product_key, dim, vector_blob, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(model_sig, product_key)
                DO UPDATE SET dim = excluded.dim, vector_blob = excluded.vector_blob, updated_at = excluded.updated_at
                """,
                rows,
            )
            conn.commit()

        logger.info("Product embedding cache upsert model=%s rows=%d", model_sig, len(rows))
