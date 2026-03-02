import logging
from typing import Any, cast

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class MilvusVectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.connected = False

    def connect(self) -> None:
        if self.connected:
            return

        logger.info("Conectando a Milvus en %s:%s", self.settings.milvus_host, self.settings.milvus_port)
        connections.connect(
            alias="default",
            host=self.settings.milvus_host,
            port=self.settings.milvus_port,
            user=self.settings.milvus_user,
            password=self.settings.milvus_password,
            db_name=self.settings.milvus_db_name,
        )
        self.connected = True

    def _collection_name(self, session_id: str, kind: str) -> str:
        short = session_id.replace("-", "")[:12]
        return f"{self.settings.milvus_collection_prefix}_{kind}_{short}"

    def collection_name(self, session_id: str, kind: str) -> str:
        return self._collection_name(session_id, kind)

    def has_collection(self, collection_name: str) -> bool:
        self.connect()
        return bool(utility.has_collection(collection_name))

    def count(self, collection_name: str) -> int:
        if not self.has_collection(collection_name):
            return 0
        collection = Collection(collection_name)
        return int(collection.num_entities)

    def recreate_collection(self, session_id: str, kind: str, dim: int) -> str:
        self.connect()
        name = self._collection_name(session_id, kind)

        if utility.has_collection(name):
            _drop_result = utility.drop_collection(name)

        fields = [
            FieldSchema(name="item_idx", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="item_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields=fields, description=f"{kind} embeddings")
        collection = Collection(name=name, schema=schema)
        _index_result = collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        collection.load()
        return name

    def insert_embeddings(
        self,
        collection_name: str,
        embeddings: np.ndarray,
        items: list[dict[str, Any]],
    ) -> None:
        collection = Collection(collection_name)
        item_indices = list(range(len(items)))
        item_ids = [item["_id"] for item in items]
        names = [item.get("nombre", "")[:1024] for item in items]
        vectors = embeddings.tolist()

        collection.insert([item_indices, item_ids, names, vectors])
        collection.flush()
        collection.load()

    def search(
        self,
        collection_name: str,
        query_vectors: np.ndarray,
        top_n: int,
    ) -> list[list[dict[str, Any]]]:
        collection = Collection(collection_name)
        collection.load()

        raw_hits = collection.search(
            data=query_vectors.tolist(),
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=top_n,
            output_fields=["item_idx", "item_id", "name"],
            _async=False,
        )
        hits = cast(Any, raw_hits)

        parsed: list[list[dict[str, Any]]] = []
        for per_query in hits:
            row: list[dict[str, Any]] = []
            for hit in per_query:
                entity = hit.entity
                row.append(
                    {
                        "item_idx": int(entity.get("item_idx")),
                        "item_id": entity.get("item_id"),
                        "name": entity.get("name"),
                        "similarity": float(hit.distance),
                    }
                )
            parsed.append(row)
        return parsed
