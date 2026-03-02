import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from app.core.config import get_settings
from app.services.milvus_client import MilvusVectorStore
from app.services.stages.base import EmbeddingStage, EnricherStage, RerankerStage, ValidatorStage
from app.services.stages.blip2_embedder import Blip2EmbeddingStage
from app.services.stages.qwen_enricher import QwenNerEnricherStage
from app.services.stages.reranker import XlmrRerankerStage
from app.services.stages.validator import QwenValidatorStage
from app.services.stages.json_parsing import normalize_attributes

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    stage: str
    seconds: float
    vram_mb: float
    details: dict[str, Any] = field(default_factory=dict)


class MatchingPipeline:
    def __init__(
        self,
        vector_store: MilvusVectorStore,
        embedder: EmbeddingStage | None = None,
        enricher: EnricherStage | None = None,
        reranker: RerankerStage | None = None,
        validator: ValidatorStage | None = None,
    ) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store
        self.embedder = embedder or Blip2EmbeddingStage(
            model_id=self.settings.blip2_model_id,
            device=self.settings.device,
            dtype=self.settings.dtype,
            offload_between_stages=self.settings.offload_between_stages,
            image_download_workers=self.settings.image_download_workers,
            image_timeout_seconds=self.settings.image_timeout_seconds,
        )
        self.enricher = enricher or QwenNerEnricherStage(
            model_id=self.settings.qwen_ner_model_id,
            device=self.settings.device,
            dtype=self.settings.dtype,
            max_new_tokens=self.settings.max_new_tokens,
            offload_between_stages=self.settings.offload_between_stages,
        )
        self.reranker = reranker or XlmrRerankerStage(
            model_id=self.settings.reranker_model_id,
            device=self.settings.device,
            dtype=self.settings.dtype,
            offload_between_stages=self.settings.offload_between_stages,
        )
        self.validator = validator or QwenValidatorStage(
            model_id=self.settings.qwen_validator_model_id,
            device=self.settings.device,
            dtype=self.settings.dtype,
            max_new_tokens=self.settings.max_new_tokens,
            offload_between_stages=self.settings.offload_between_stages,
        )
        self._cache_dir = self.settings.data_dir / "checkpoints"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _compact_attrs(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return normalize_attributes(value, raw_fallback=str(value.get("raw", "")))
        return normalize_attributes({}, raw_fallback=None)

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _text_tokens(self, value: str) -> set[str]:
        return {tok for tok in self._normalize_text(value).split() if len(tok) > 1}

    def _token_overlap(self, a: str, b: str) -> float:
        left = self._text_tokens(a)
        right = self._text_tokens(b)
        if not left or not right:
            return 0.0
        return float(len(left & right) / max(1, min(len(left), len(right))))

    def _attrs_for_validator_prompt(self, attrs: dict[str, Any]) -> dict[str, Any]:
        clean = self._compact_attrs(attrs)
        return {
            "marca": clean.get("marca"),
            "modelo": clean.get("modelo"),
            "categoria": clean.get("categoria"),
            "unidad": clean.get("unidad"),
            "atributos": (clean.get("atributos") or [])[:5],
        }

    def _fast_validator_decision(
        self,
        anchor_name: str,
        candidate_name: str,
        anchor_attrs: dict[str, Any],
        candidate_attrs: dict[str, Any],
        similarity: float,
        reranker_score: float,
    ) -> dict[str, Any] | None:
        anchor_brand = (anchor_attrs.get("marca") or "").strip().lower()
        candidate_brand = (candidate_attrs.get("marca") or "").strip().lower()
        anchor_unit = (anchor_attrs.get("unidad") or "").strip().lower()
        candidate_unit = (candidate_attrs.get("unidad") or "").strip().lower()

        if anchor_brand and candidate_brand and anchor_brand != candidate_brand:
            return {
                "validation_score": 0.05,
                "review_flag": True,
                "reason": "brand_mismatch_fast_rule",
            }

        if anchor_unit and candidate_unit and anchor_unit != candidate_unit:
            return {
                "validation_score": 0.20,
                "review_flag": True,
                "reason": "unit_mismatch_fast_rule",
            }

        anchor_norm = self._normalize_text(anchor_name)
        candidate_norm = self._normalize_text(candidate_name)
        overlap = self._token_overlap(anchor_name, candidate_name)

        if (
            similarity >= 0.95
            and overlap >= 0.60
            and (not anchor_brand or not candidate_brand or anchor_brand == candidate_brand)
            and (not anchor_unit or not candidate_unit or anchor_unit == candidate_unit)
        ):
            return {
                "validation_score": min(0.99, 0.80 + 0.19 * similarity),
                "review_flag": False,
                "reason": "high_similarity_fast_rule",
            }

        if similarity >= 0.92 and overlap >= 0.45 and reranker_score >= 0.25:
            return {
                "validation_score": min(0.95, 0.70 + 0.25 * similarity),
                "review_flag": False,
                "reason": "overlap_fast_rule",
            }

        if anchor_norm and anchor_norm in candidate_norm and reranker_score >= 0.92:
            return {
                "validation_score": min(0.98, 0.72 + 0.28 * reranker_score),
                "review_flag": False,
                "reason": "high_confidence_fast_rule",
            }

        if reranker_score <= 0.22 and similarity <= 0.70:
            return {
                "validation_score": 0.08,
                "review_flag": True,
                "reason": "low_confidence_fast_rule",
            }

        return None

    def _vram_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    def _run_stage(self, name: str, fn, details: dict[str, Any] | None = None):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        logger.info("[Pipeline] Iniciando etapa: %s", name)
        started = time.perf_counter()
        output = fn()
        elapsed = time.perf_counter() - started
        vram = self._vram_mb()
        logger.info("[Pipeline] Etapa %s completada en %.2fs (VRAM pico %.1f MB)", name, elapsed, vram)

        metric = Metric(stage=name, seconds=elapsed, vram_mb=vram, details=details or {})
        return output, metric

    def _run_cached_stage(
        self,
        stage_name: str,
        read_cache: bool,
        details: dict[str, Any],
        load_fn: Callable[[], Any | None],
        compute_fn: Callable[[], Any],
        save_fn: Callable[[Any], None] | None = None,
    ):
        if read_cache:
            started = time.perf_counter()
            cached = load_fn()
            if cached is not None:
                elapsed = time.perf_counter() - started
                metric = Metric(
                    stage=stage_name,
                    seconds=elapsed,
                    vram_mb=self._vram_mb(),
                    details={**details, "cache_hit": True},
                )
                logger.info("[Pipeline] Etapa %s recuperada de cache", stage_name)
                return cached, metric

        output, metric = self._run_stage(stage_name, compute_fn, details={**details, "cache_hit": False})
        if save_fn is not None:
            try:
                save_fn(output)
            except Exception as exc:
                logger.warning("No se pudo persistir cache en %s: %s", stage_name, exc)
        return output, metric

    def _prepare_session_items(self, session_data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        anchors = [dict(item) for item in session_data["anchors"]]
        products = [dict(item) for item in session_data["products"]]
        for idx, item in enumerate(anchors):
            item["_id"] = f"a_{idx}"
        for idx, item in enumerate(products):
            item["_id"] = f"p_{idx}"
        return anchors, products

    def _cache_key(self, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:14]

    def _session_cache_dir(self, session_id: str) -> Path:
        path = self._cache_dir / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _cache_path(self, session_id: str, stage: str, key: str, ext: str) -> Path:
        return self._session_cache_dir(session_id) / f"{stage}_{key}.{ext}"

    def _load_json_cache(self, session_id: str, stage: str, key: str) -> Any | None:
        path = self._cache_path(session_id, stage, key, "json")
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_json_cache(self, session_id: str, stage: str, key: str, payload: Any) -> None:
        path = self._cache_path(session_id, stage, key, "json")
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _load_npy_cache(self, session_id: str, stage: str, key: str) -> np.ndarray | None:
        path = self._cache_path(session_id, stage, key, "npy")
        if not path.exists():
            return None
        return np.load(path)

    def _save_npy_cache(self, session_id: str, stage: str, key: str, payload: np.ndarray) -> None:
        path = self._cache_path(session_id, stage, key, "npy")
        np.save(path, payload)

    def _ensure_milvus_collections(
        self,
        session_id: str,
        anchors: list[dict[str, Any]],
        products: list[dict[str, Any]],
        anchor_embeddings: np.ndarray,
        product_embeddings: np.ndarray,
    ) -> str:
        products_collection = self.vector_store.collection_name(session_id, "products")
        anchors_collection = self.vector_store.collection_name(session_id, "anchors")
        dim = int(anchor_embeddings.shape[1])

        products_ok = self.vector_store.has_collection(products_collection) and (
            self.vector_store.count(products_collection) == len(products)
        )
        anchors_ok = self.vector_store.has_collection(anchors_collection) and (
            self.vector_store.count(anchors_collection) == len(anchors)
        )

        if not anchors_ok:
            anchors_collection = self.vector_store.recreate_collection(session_id, "anchors", dim)
            self.vector_store.insert_embeddings(anchors_collection, anchor_embeddings, anchors)

        if not products_ok:
            products_collection = self.vector_store.recreate_collection(session_id, "products", dim)
            self.vector_store.insert_embeddings(products_collection, product_embeddings, products)

        return products_collection

    def process(
        self,
        session_data: dict[str, Any],
        top_n: int,
        top_k: int,
        batch_size: int | None = None,
        ner_batch_size: int | None = None,
        validator_batch_size: int | None = None,
        use_resume: bool = True,
        extraction_prompt: str | None = None,
        validation_prompt: str | None = None,
        extraction_prompt_id: str = "default",
        validation_prompt_id: str = "default",
    ) -> dict[str, Any]:
        effective_batch = batch_size or self.settings.batch_size
        effective_ner_batch = ner_batch_size or max(1, effective_batch // 2)
        effective_validator_batch = validator_batch_size or max(1, effective_batch // 2)
        metrics: list[Metric] = []
        session_id = session_data["session_id"]
        extraction_prompt_sig = hashlib.sha1((extraction_prompt or "").encode("utf-8")).hexdigest()[:10]
        validation_prompt_sig = hashlib.sha1((validation_prompt or "").encode("utf-8")).hexdigest()[:10]

        anchors, products = self._prepare_session_items(session_data)
        product_by_id = {item["_id"]: item for item in products}

        embed_key = self._cache_key(
            {
                "stage": "embed",
                "version": "v2",
                "blip2": self.settings.blip2_model_id,
                "dtype": self.settings.dtype,
                "device": self.settings.device,
                "anchors": len(anchors),
                "products": len(products),
            }
        )

        def load_embeddings_cached() -> dict[str, Any] | None:
            anchor_embeddings = self._load_npy_cache(session_id, "anchor_embeddings", embed_key)
            product_embeddings = self._load_npy_cache(session_id, "product_embeddings", embed_key)
            if anchor_embeddings is None or product_embeddings is None:
                return None
            products_collection = self._ensure_milvus_collections(
                session_id,
                anchors,
                products,
                anchor_embeddings,
                product_embeddings,
            )
            return {
                "anchor_embeddings": anchor_embeddings,
                "product_embeddings": product_embeddings,
                "products_collection": products_collection,
            }

        def compute_embeddings() -> dict[str, Any]:
            all_items = anchors + products
            all_embeddings = self.embedder.embed_records(all_items, effective_batch)
            anchor_embeddings = all_embeddings[: len(anchors)]
            product_embeddings = all_embeddings[len(anchors) :]
            products_collection = self._ensure_milvus_collections(
                session_id,
                anchors,
                products,
                anchor_embeddings,
                product_embeddings,
            )
            return {
                "anchor_embeddings": anchor_embeddings,
                "product_embeddings": product_embeddings,
                "products_collection": products_collection,
            }

        def save_embeddings_cached(payload: dict[str, Any]) -> None:
            self._save_npy_cache(session_id, "anchor_embeddings", embed_key, payload["anchor_embeddings"])
            self._save_npy_cache(session_id, "product_embeddings", embed_key, payload["product_embeddings"])

        embeddings_payload, metric = self._run_cached_stage(
            "embeddings_and_milvus",
            read_cache=use_resume,
            details={"anchors": len(anchors), "products": len(products), "batch_size": effective_batch},
            load_fn=load_embeddings_cached,
            compute_fn=compute_embeddings,
            save_fn=save_embeddings_cached,
        )
        metrics.append(metric)

        anchor_embeddings = embeddings_payload["anchor_embeddings"]
        products_collection = embeddings_payload["products_collection"]

        search_key = self._cache_key(
            {
                "stage": "search",
                "version": "v2",
                "top_n": top_n,
                "session": session_id,
                "model": self.settings.blip2_model_id,
            }
        )

        def load_search_cached():
            return self._load_json_cache(session_id, "vector_search", search_key)

        def compute_search():
            return self.vector_store.search(
                collection_name=products_collection,
                query_vectors=anchor_embeddings,
                top_n=top_n,
            )

        def save_search_cached(payload: Any) -> None:
            self._save_json_cache(session_id, "vector_search", search_key, payload)

        search_results, metric = self._run_cached_stage(
            "vector_search",
            read_cache=use_resume,
            details={"top_n": top_n, "queries": len(anchors), "candidates": len(anchors) * top_n},
            load_fn=load_search_cached,
            compute_fn=compute_search,
            save_fn=save_search_cached,
        )
        metrics.append(metric)

        candidate_ids = {item["item_id"] for row in search_results for item in row}
        candidate_items = [product_by_id[candidate_id] for candidate_id in candidate_ids if candidate_id in product_by_id]

        ner_key = self._cache_key(
            {
                "stage": "ner",
                "version": "v3",
                "top_n": top_n,
                "prompt": extraction_prompt_id,
                "prompt_sig": extraction_prompt_sig,
                "model": self.settings.qwen_ner_model_id,
                "items": len(anchors) + len(candidate_items),
            }
        )

        def load_ner_cached():
            return self._load_json_cache(session_id, "qwen_ner", ner_key)

        def compute_ner():
            all_for_ner = anchors + candidate_items
            return self.enricher.extract_attributes(
                all_for_ner,
                effective_ner_batch,
                prompt_template=extraction_prompt,
            )

        def save_ner_cached(payload: Any) -> None:
            self._save_json_cache(session_id, "qwen_ner", ner_key, payload)

        attributes_by_id, metric = self._run_cached_stage(
            "qwen_ner",
            read_cache=use_resume,
            details={
                "items": len(anchors) + len(candidate_items),
                "batch_size": effective_ner_batch,
                "prompt": extraction_prompt_id,
            },
            load_fn=load_ner_cached,
            compute_fn=compute_ner,
            save_fn=save_ner_cached,
        )
        metrics.append(metric)

        compact_attrs_by_id = {item_id: self._compact_attrs(attrs) for item_id, attrs in attributes_by_id.items()}

        base_pairs: list[dict[str, Any]] = []
        for anchor_idx, anchor in enumerate(anchors):
            for hit in search_results[anchor_idx]:
                candidate = product_by_id[hit["item_id"]]
                base_pairs.append(
                    {
                        "anchor_id": anchor["_id"],
                        "candidate_id": candidate["_id"],
                        "anchor_text": anchor["nombre"],
                        "candidate_text": candidate["nombre"],
                        "similarity": hit["similarity"],
                    }
                )

        reranker_key = self._cache_key(
            {
                "stage": "rerank",
                "version": "v3",
                "top_n": top_n,
                "model": self.settings.reranker_model_id,
                "pairs": len(base_pairs),
            }
        )

        def load_rerank_cached():
            return self._load_json_cache(session_id, "xlmr_reranker", reranker_key)

        def compute_rerank():
            rerank_scores = self.reranker.score_pairs(base_pairs, effective_batch)
            scored_pairs = []
            for pair, score in zip(base_pairs, rerank_scores):
                row = dict(pair)
                row["reranker_score"] = float(score)
                row["combined_score"] = (0.25 * float(score)) + (0.75 * float(pair["similarity"]))
                scored_pairs.append(row)
            return scored_pairs

        def save_rerank_cached(payload: Any) -> None:
            self._save_json_cache(session_id, "xlmr_reranker", reranker_key, payload)

        rerank_pairs, metric = self._run_cached_stage(
            "xlmr_reranker",
            read_cache=use_resume,
            details={"pairs": len(base_pairs), "batch_size": effective_batch},
            load_fn=load_rerank_cached,
            compute_fn=compute_rerank,
            save_fn=save_rerank_cached,
        )
        metrics.append(metric)

        grouped: dict[str, list[dict[str, Any]]] = {anchor["_id"]: [] for anchor in anchors}
        for pair in rerank_pairs:
            grouped[pair["anchor_id"]].append(pair)

        for anchor_id, rows in grouped.items():
            grouped[anchor_id] = sorted(rows, key=lambda value: value["combined_score"], reverse=True)

        validation_pairs: list[dict[str, Any]] = []
        validation_map: list[tuple[str, str]] = []
        validation_lookup: dict[str, dict[str, Any]] = {}
        for anchor in anchors:
            shortlisted = grouped[anchor["_id"]][:top_k]
            for row in shortlisted:
                candidate = product_by_id[row["candidate_id"]]
                anchor_attrs = compact_attrs_by_id.get(anchor["_id"], {})
                candidate_attrs = compact_attrs_by_id.get(candidate["_id"], {})
                prompt_anchor_attrs = self._attrs_for_validator_prompt(anchor_attrs)
                prompt_candidate_attrs = self._attrs_for_validator_prompt(candidate_attrs)
                fast = self._fast_validator_decision(
                    anchor_name=anchor["nombre"],
                    candidate_name=candidate["nombre"],
                    anchor_attrs=prompt_anchor_attrs,
                    candidate_attrs=prompt_candidate_attrs,
                    similarity=float(row["similarity"]),
                    reranker_score=float(row["reranker_score"]),
                )
                key = f"{anchor['_id']}|{candidate['_id']}"
                if fast is not None:
                    validation_lookup[key] = fast
                    continue

                validation_pairs.append(
                    {
                        "anchor_name": anchor["nombre"],
                        "anchor_attrs": prompt_anchor_attrs,
                        "candidate_name": candidate["nombre"],
                        "candidate_attrs": prompt_candidate_attrs,
                        "url": candidate.get("url_producto"),
                        "price": candidate.get("precioFinal"),
                    }
                )
                validation_map.append((anchor["_id"], candidate["_id"]))

        validator_key = self._cache_key(
            {
                "stage": "validator",
                "version": "v3",
                "top_n": top_n,
                "top_k": top_k,
                "prompt": validation_prompt_id,
                "prompt_sig": validation_prompt_sig,
                "model": self.settings.qwen_validator_model_id,
                "pairs": len(validation_pairs),
                "fast_resolved": len(validation_lookup),
            }
        )

        def load_validator_cached():
            return self._load_json_cache(session_id, "qwen_validator", validator_key)

        def compute_validator():
            return self.validator.validate_pairs(
                validation_pairs,
                effective_validator_batch,
                prompt_template=validation_prompt,
            )

        def save_validator_cached(payload: Any) -> None:
            self._save_json_cache(session_id, "qwen_validator", validator_key, payload)

        validations, metric = self._run_cached_stage(
            "qwen_validator",
            read_cache=use_resume and len(validation_pairs) > 0,
            details={
                "pairs": len(validation_pairs),
                "fast_resolved": len(validation_lookup),
                "top_k": top_k,
                "batch_size": effective_validator_batch,
                "prompt": validation_prompt_id,
            },
            load_fn=load_validator_cached,
            compute_fn=compute_validator,
            save_fn=save_validator_cached if len(validation_pairs) > 0 else None,
        )
        metrics.append(metric)

        validation_lookup.update(
            {
                f"{anchor_id}|{candidate_id}": validation
                for (anchor_id, candidate_id), validation in zip(validation_map, validations)
            }
        )

        final_results = []
        for anchor in anchors:
            anchor_rows = grouped[anchor["_id"]][:top_k]
            matches = []
            for row in anchor_rows:
                candidate = product_by_id[row["candidate_id"]]
                validation = validation_lookup.get(f"{anchor['_id']}|{candidate['_id']}", {})
                fallback_score = max(0.0, min(1.0, (0.5 * float(row["reranker_score"])) + (0.5 * float(row["similarity"]))))
                matches.append(
                    {
                        "nombre": candidate.get("nombre"),
                        "url": candidate.get("url_producto"),
                        "img": candidate.get("img"),
                        "precio": float(candidate.get("precioFinal", 0.0)),
                        "score_similitud": float(row["similarity"]),
                        "score_reranker": float(row["reranker_score"]),
                        "score_validacion": float(validation.get("validation_score", fallback_score)),
                        "revisar": bool(validation.get("review_flag", fallback_score < 0.66)),
                        "atributos": compact_attrs_by_id.get(candidate["_id"], {}),
                        "sitio": candidate.get("sitio"),
                        "seller": candidate.get("seller"),
                    }
                )

            final_results.append(
                {
                    "anchor_id": anchor["_id"],
                    "anchor_nombre": anchor["nombre"],
                    "atributos_anchor": compact_attrs_by_id.get(anchor["_id"], {}),
                    "matches": matches,
                }
            )

        return {
            "session_id": session_id,
            "top_n": top_n,
            "top_k": top_k,
            "use_resume": use_resume,
            "extraction_prompt_id": extraction_prompt_id,
            "validation_prompt_id": validation_prompt_id,
            "results": final_results,
            "metrics": [metric.__dict__ for metric in metrics],
        }
