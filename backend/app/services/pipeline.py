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
from app.services.product_embedding_cache import ProductEmbeddingCache, product_cache_key
from app.services.stages.base import EmbeddingStage, EnricherStage, RerankerStage, ValidatorStage
from app.services.stages.blip2_embedder import Blip2EmbeddingStage
from app.services.stages.qwen_enricher import QwenNerEnricherStage
from app.services.stages.reranker import XlmrRerankerStage, build_pair_text
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
            model_id=self.settings.vllm_ner_model_id if self.settings.use_vllm else self.settings.qwen_ner_model_id,
            device=self.settings.device,
            dtype=self.settings.dtype,
            max_new_tokens=self.settings.ner_max_new_tokens,
            use_vllm=self.settings.use_vllm,
            vllm_base_url=self.settings.vllm_base_url,
            vllm_api_key=self.settings.vllm_api_key,
            vllm_timeout_seconds=self.settings.vllm_timeout_seconds,
            vllm_max_retries=self.settings.vllm_max_retries,
            vllm_disable_thinking=self.settings.vllm_disable_thinking,
            vllm_max_parallel=self.settings.vllm_max_parallel,
            offload_between_stages=self.settings.offload_between_stages,
            strict_json=True,
        )
        self.reranker = reranker or XlmrRerankerStage(
            model_id=self.settings.reranker_model_id,
            device=self.settings.device,
            dtype=self.settings.dtype,
            offload_between_stages=self.settings.offload_between_stages,
        )
        self.validator = validator or QwenValidatorStage(
            model_id=self.settings.vllm_validator_model_id if self.settings.use_vllm else self.settings.qwen_validator_model_id,
            device=self.settings.device,
            dtype=self.settings.dtype,
            max_new_tokens=self.settings.validator_max_new_tokens,
            use_vllm=self.settings.use_vllm,
            vllm_base_url=self.settings.vllm_base_url,
            vllm_api_key=self.settings.vllm_api_key,
            vllm_timeout_seconds=self.settings.vllm_timeout_seconds,
            vllm_max_retries=self.settings.vllm_max_retries,
            vllm_disable_thinking=self.settings.vllm_disable_thinking,
            vllm_max_parallel=self.settings.vllm_max_parallel,
            vllm_context_window=self.settings.vllm_context_window,
            vllm_context_reserve=self.settings.vllm_context_reserve,
            offload_between_stages=self.settings.offload_between_stages,
        )
        self._cache_dir = self.settings.data_dir / "checkpoints"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._global_embedding_cache = ProductEmbeddingCache(self.settings.data_dir / "global_product_embeddings.sqlite3")

    def _embedding_model_sig(self) -> str:
        raw = {
            "model": self.settings.blip2_model_id,
            "dtype": self.settings.dtype,
            "device": self.settings.device,
            "version": "global_v1",
        }
        return hashlib.sha1(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def _embed_products_with_global_cache(
        self,
        products: list[dict[str, Any]],
        batch_size: int,
    ) -> tuple[np.ndarray, dict[str, int]]:
        if not products:
            return np.zeros((0, 768), dtype=np.float32), {"hits": 0, "misses": 0}

        model_sig = self._embedding_model_sig()
        keys = [product_cache_key(item) for item in products]
        unique_keys = list(dict.fromkeys(keys))
        cached = self._global_embedding_cache.get_many(model_sig, unique_keys)

        miss_key_to_item: dict[str, dict[str, Any]] = {}
        for key, item in zip(keys, products):
            if key not in cached and key not in miss_key_to_item:
                miss_key_to_item[key] = item

        misses = list(miss_key_to_item.items())
        if misses:
            miss_items = [item for _, item in misses]
            miss_embeddings = self.embedder.embed_records(miss_items, batch_size)
            to_cache = {
                key: miss_embeddings[idx]
                for idx, (key, _) in enumerate(misses)
            }
            self._global_embedding_cache.put_many(model_sig, to_cache)
            cached.update(to_cache)

        ordered = np.stack([cached[key] for key in keys], axis=0).astype(np.float32)
        stats = {"hits": len(keys) - len(misses), "misses": len(misses)}
        return ordered, stats

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
            "brand": clean.get("brand"),
            "category": clean.get("category"),
        }

    def _local_vector_search(
        self,
        anchor_embeddings: np.ndarray,
        product_embeddings: np.ndarray,
        products: list[dict[str, Any]],
        top_n: int,
    ) -> list[list[dict[str, Any]]]:
        logger.warning("[Pipeline] Milvus no disponible, usando busqueda vectorial local (numpy)")
        if len(products) == 0:
            return [[] for _ in range(len(anchor_embeddings))]

        k = max(1, min(top_n, len(products)))
        sims = np.matmul(anchor_embeddings.astype(np.float32), product_embeddings.astype(np.float32).T)
        results: list[list[dict[str, Any]]] = []

        for row in sims:
            idxs = np.argpartition(-row, k - 1)[:k]
            idxs = idxs[np.argsort(-row[idxs])]
            parsed_row: list[dict[str, Any]] = []
            for idx in idxs.tolist():
                product = products[idx]
                parsed_row.append(
                    {
                        "item_idx": int(idx),
                        "item_id": product["_id"],
                        "name": product.get("nombre", ""),
                        "similarity": float(row[idx]),
                    }
                )
            results.append(parsed_row)
        return results

    def _fast_validator_decision(
        self,
        anchor_name: str,
        candidate_name: str,
        anchor_attrs: dict[str, Any],
        candidate_attrs: dict[str, Any],
        similarity: float,
        reranker_score: float,
    ) -> dict[str, Any] | None:
        anchor_brand = (anchor_attrs.get("brand") or "").strip().lower()
        candidate_brand = (candidate_attrs.get("brand") or "").strip().lower()
        anchor_category = (anchor_attrs.get("category") or "").strip().lower()
        candidate_category = (candidate_attrs.get("category") or "").strip().lower()

        if anchor_brand and candidate_brand and anchor_brand != candidate_brand:
            return {
                "validation_score": 0.05,
                "review_flag": True,
                "reason": "brand_mismatch_fast_rule",
            }

        if anchor_category and candidate_category and anchor_category != candidate_category:
            return {
                "validation_score": 0.12,
                "review_flag": True,
                "reason": "category_mismatch_fast_rule",
            }

        anchor_norm = self._normalize_text(anchor_name)
        candidate_norm = self._normalize_text(candidate_name)
        overlap = self._token_overlap(anchor_name, candidate_name)

        if (
            similarity >= 0.95
            and overlap >= 0.60
            and (not anchor_brand or not candidate_brand or anchor_brand == candidate_brand)
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

    def _work_unit_label(self, stage: str) -> str | None:
        mapping = {
            "embeddings_and_milvus": "records",
            "vector_search": "candidates",
            "qwen_ner": "items",
            "xlmr_reranker": "pairs",
            "qwen_validator": "candidates",
        }
        return mapping.get(stage)

    def _work_units(self, stage: str, details: dict[str, Any]) -> float | None:
        if stage == "embeddings_and_milvus":
            anchors = int(details.get("anchors", 0))
            products = int(details.get("products", 0))
            return float(anchors + products)
        if stage == "vector_search":
            return float(int(details.get("candidates", 0)))
        if stage == "qwen_ner":
            return float(int(details.get("items", 0)))
        if stage == "xlmr_reranker":
            return float(int(details.get("pairs", 0)))
        if stage == "qwen_validator":
            groups = int(details.get("groups", 0))
            fast_resolved = int(details.get("fast_resolved", 0))
            return float(groups + fast_resolved)
        return None

    def _attach_metric_efficiency(self, metrics: list[Metric]) -> None:
        total_stage_seconds = sum(metric.seconds for metric in metrics if metric.seconds > 0.0)
        for metric in metrics:
            details = metric.details
            units = self._work_units(metric.stage, details)
            unit_label = self._work_unit_label(metric.stage)
            if units is not None:
                details["work_units"] = int(units) if float(units).is_integer() else units
            if unit_label:
                details["work_unit_label"] = unit_label

            if units and metric.seconds > 0.0:
                details["throughput_per_sec"] = float(units / metric.seconds)
                details["ms_per_unit"] = float((metric.seconds * 1000.0) / units)
            else:
                details["throughput_per_sec"] = None
                details["ms_per_unit"] = None

            if total_stage_seconds > 0.0:
                details["time_share_pct"] = float((metric.seconds / total_stage_seconds) * 100.0)
            else:
                details["time_share_pct"] = None

            hits = int(details.get("product_cache_hits", 0))
            misses = int(details.get("product_cache_misses", 0))
            total = hits + misses
            if total > 0:
                details["product_cache_hit_rate_pct"] = float((hits / total) * 100.0)

            if metric.stage == "qwen_validator":
                groups = int(details.get("groups", 0))
                fast_resolved = int(details.get("fast_resolved", 0))
                total_candidates = groups + fast_resolved
                if total_candidates > 0:
                    details["llm_usage_pct"] = float((groups / total_candidates) * 100.0)
                    details["llm_avoidance_pct"] = float((fast_resolved / total_candidates) * 100.0)

    def _build_efficiency_stats(
        self,
        metrics: list[Metric],
        wall_seconds: float,
        anchors_count: int,
        products_count: int,
        top_n: int,
        top_k: int,
        candidate_items_count: int,
        unresolved_candidates: int,
        fast_resolved: int,
    ) -> dict[str, Any]:
        stage_seconds = sum(metric.seconds for metric in metrics)
        cached_stages = sum(1 for metric in metrics if bool(metric.details.get("cache_hit")))
        dominant_stage = max(metrics, key=lambda metric: metric.seconds).stage if metrics else None

        vector_candidates = anchors_count * top_n
        shortlisted_candidates = anchors_count * top_k
        rerank_pruning_pct = None
        if vector_candidates > 0:
            rerank_pruning_pct = float((1.0 - (shortlisted_candidates / vector_candidates)) * 100.0)

        total_validator_candidates = unresolved_candidates + fast_resolved
        validator_llm_usage_pct = None
        validator_llm_avoidance_pct = None
        if total_validator_candidates > 0:
            validator_llm_usage_pct = float((unresolved_candidates / total_validator_candidates) * 100.0)
            validator_llm_avoidance_pct = float((fast_resolved / total_validator_candidates) * 100.0)

        return {
            "wall_seconds": float(wall_seconds),
            "stage_seconds": float(stage_seconds),
            "stage_overhead_seconds": float(max(0.0, wall_seconds - stage_seconds)),
            "stages_total": len(metrics),
            "stages_cached": cached_stages,
            "stage_cache_hit_rate_pct": float((cached_stages / len(metrics)) * 100.0) if metrics else 0.0,
            "dominant_stage": dominant_stage,
            "anchors": anchors_count,
            "products": products_count,
            "candidate_products_considered": candidate_items_count,
            "vector_candidates": vector_candidates,
            "shortlisted_candidates": shortlisted_candidates,
            "rerank_pruning_pct": rerank_pruning_pct,
            "validator_candidates": total_validator_candidates,
            "validator_llm_calls": unresolved_candidates,
            "validator_fast_resolved": fast_resolved,
            "validator_llm_usage_pct": validator_llm_usage_pct,
            "validator_llm_avoidance_pct": validator_llm_avoidance_pct,
        }

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

    def _load_latest_stage_json_cache(self, session_id: str, stage: str) -> Any | None:
        session_dir = self._session_cache_dir(session_id)
        candidates = sorted(
            session_dir.glob(f"{stage}_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
        return None

    def _save_json_cache(self, session_id: str, stage: str, key: str, payload: Any) -> None:
        path = self._cache_path(session_id, stage, key, "json")
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _normalize_search_cache(
        self,
        payload: Any,
        expected_queries: int,
        top_n: int,
    ) -> list[list[dict[str, Any]]] | None:
        if not isinstance(payload, list) or len(payload) != expected_queries:
            return None

        normalized: list[list[dict[str, Any]]] = []
        for row in payload:
            if not isinstance(row, list):
                return None
            clean_row: list[dict[str, Any]] = []
            for item in row:
                if not isinstance(item, dict):
                    continue
                if "item_id" not in item or "similarity" not in item:
                    continue
                clean_row.append(item)
            if len(clean_row) < top_n:
                return None
            normalized.append(clean_row[:top_n])
        return normalized

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
        use_fast_rules: bool = True,
        extraction_prompt: str | None = None,
        validation_prompt: str | None = None,
        extraction_prompt_id: str = "default",
        validation_prompt_id: str = "default",
        th_accept: float | None = None,
        th_reject: float | None = None,
    ) -> dict[str, Any]:
        process_started = time.perf_counter()
        effective_batch = batch_size or self.settings.batch_size
        effective_ner_batch = ner_batch_size or max(1, effective_batch // 2)
        effective_validator_batch = validator_batch_size or max(1, effective_batch // 2)
        effective_th_accept = float(self.settings.th_accept if th_accept is None else th_accept)
        effective_th_reject = float(self.settings.th_reject if th_reject is None else th_reject)
        metrics: list[Metric] = []
        session_id = session_data["session_id"]
        extraction_prompt_sig = hashlib.sha1((extraction_prompt or "").encode("utf-8")).hexdigest()[:10]
        validation_prompt_sig = hashlib.sha1((validation_prompt or "").encode("utf-8")).hexdigest()[:10]

        anchors, products = self._prepare_session_items(session_data)
        product_by_id = {item["_id"]: item for item in products}

        search_key = self._cache_key(
            {
                "stage": "search",
                "version": "v2",
                "top_n": top_n,
                "session": session_id,
                "model": self.settings.blip2_model_id,
            }
        )

        search_results = self._load_json_cache(session_id, "vector_search", search_key) if use_resume else None
        search_results = self._normalize_search_cache(search_results, expected_queries=len(anchors), top_n=top_n)
        if search_results is None and use_resume:
            legacy_search = self._load_latest_stage_json_cache(session_id, "vector_search")
            legacy_search = self._normalize_search_cache(legacy_search, expected_queries=len(anchors), top_n=top_n)
            if legacy_search is not None:
                search_results = legacy_search
                self._save_json_cache(session_id, "vector_search", search_key, search_results)
                logger.info("[Pipeline] Reutilizando cache legacy de vector_search para session=%s", session_id)
        if search_results is not None:
            metrics.append(
                Metric(
                    stage="embeddings_and_milvus",
                    seconds=0.0,
                    vram_mb=self._vram_mb(),
                    details={
                        "anchors": len(anchors),
                        "products": len(products),
                        "batch_size": effective_batch,
                        "cache_hit": True,
                        "skipped": True,
                        "reason": "vector_search_cache",
                    },
                )
            )
            metrics.append(
                Metric(
                    stage="vector_search",
                    seconds=0.0,
                    vram_mb=self._vram_mb(),
                    details={
                        "top_n": top_n,
                        "queries": len(anchors),
                        "candidates": len(anchors) * top_n,
                        "cache_hit": True,
                    },
                )
            )

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
            products_collection: str | None = None
            milvus_ready = False
            try:
                products_collection = self._ensure_milvus_collections(
                    session_id,
                    anchors,
                    products,
                    anchor_embeddings,
                    product_embeddings,
                )
                milvus_ready = True
            except Exception as exc:
                logger.warning("[Pipeline] No se pudo rehidratar colecciones Milvus desde cache: %s", exc)
            return {
                "anchor_embeddings": anchor_embeddings,
                "product_embeddings": product_embeddings,
                "products_collection": products_collection,
                "milvus_ready": milvus_ready,
            }

        def compute_embeddings() -> dict[str, Any]:
            anchor_embeddings = self.embedder.embed_records(anchors, effective_batch)
            product_embeddings, product_cache_stats = self._embed_products_with_global_cache(products, effective_batch)
            products_collection: str | None = None
            milvus_ready = False
            try:
                products_collection = self._ensure_milvus_collections(
                    session_id,
                    anchors,
                    products,
                    anchor_embeddings,
                    product_embeddings,
                )
                milvus_ready = True
            except Exception as exc:
                logger.warning("[Pipeline] Milvus no disponible luego de embeddings: %s", exc)
            return {
                "anchor_embeddings": anchor_embeddings,
                "product_embeddings": product_embeddings,
                "products_collection": products_collection,
                "milvus_ready": milvus_ready,
                "product_cache": product_cache_stats,
            }

        def save_embeddings_cached(payload: dict[str, Any]) -> None:
            self._save_npy_cache(session_id, "anchor_embeddings", embed_key, payload["anchor_embeddings"])
            self._save_npy_cache(session_id, "product_embeddings", embed_key, payload["product_embeddings"])

        if search_results is None:
            embeddings_payload, metric = self._run_cached_stage(
                "embeddings_and_milvus",
                read_cache=use_resume,
                details={"anchors": len(anchors), "products": len(products), "batch_size": effective_batch},
                load_fn=load_embeddings_cached,
                compute_fn=compute_embeddings,
                save_fn=save_embeddings_cached,
            )
            product_cache_stats = embeddings_payload.get("product_cache")
            if isinstance(product_cache_stats, dict):
                metric.details["product_cache_hits"] = int(product_cache_stats.get("hits", 0))
                metric.details["product_cache_misses"] = int(product_cache_stats.get("misses", 0))
            metrics.append(metric)

            anchor_embeddings = embeddings_payload["anchor_embeddings"]
            product_embeddings = embeddings_payload["product_embeddings"]
            products_collection = embeddings_payload["products_collection"]
            milvus_ready = bool(embeddings_payload.get("milvus_ready", False))

            def load_search_cached():
                return self._load_json_cache(session_id, "vector_search", search_key)

            def compute_search():
                if milvus_ready and products_collection:
                    try:
                        return self.vector_store.search(
                            collection_name=products_collection,
                            query_vectors=anchor_embeddings,
                            top_n=top_n,
                        )
                    except Exception as exc:
                        logger.warning("[Pipeline] Fallo busqueda Milvus, fallback local: %s", exc)
                return self._local_vector_search(
                    anchor_embeddings=anchor_embeddings,
                    product_embeddings=product_embeddings,
                    products=products,
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
                "version": "v4",
                "top_n": top_n,
                "prompt": extraction_prompt_id,
                "prompt_sig": extraction_prompt_sig,
                "model": self.settings.vllm_ner_model_id if self.settings.use_vllm else self.settings.qwen_ner_model_id,
                "backend": "vllm" if self.settings.use_vllm else "transformers",
                "items": len(anchors) + len(candidate_items),
                "text_sig": hashlib.sha1(
                    "\n".join(
                        sorted(self._normalize_text(item.get("nombre", "")) for item in (anchors + candidate_items))
                    ).encode("utf-8")
                ).hexdigest()[:12],
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
        metric.details["parse_fail_count"] = int(getattr(self.enricher, "parse_fail_count", 0))
        ner_total = max(1, int(getattr(self.enricher, "total_count", len(anchors) + len(candidate_items))))
        metric.details["parse_fail_rate_pct"] = float((metric.details["parse_fail_count"] / ner_total) * 100.0)
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
                        "anchor_ner": compact_attrs_by_id.get(anchor["_id"], {}),
                        "candidate_ner": compact_attrs_by_id.get(candidate["_id"], {}),
                    }
                )

        for pair in base_pairs:
            text_a, text_b = build_pair_text(
                anchor_text=pair["anchor_text"],
                candidate_text=pair["candidate_text"],
                anchor_ner=pair.get("anchor_ner", {}),
                candidate_ner=pair.get("candidate_ner", {}),
            )
            pair["text_a"] = text_a
            pair["text_b"] = text_b

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

        validation_groups: list[dict[str, Any]] = []
        validation_lookup: dict[str, dict[str, Any]] = {}
        unresolved_candidates = 0
        for anchor in anchors:
            shortlisted = grouped[anchor["_id"]][:top_k]
            anchor_prompt_attrs = self._attrs_for_validator_prompt(compact_attrs_by_id.get(anchor["_id"], {}))
            for row in shortlisted:
                candidate = product_by_id[row["candidate_id"]]
                anchor_attrs = compact_attrs_by_id.get(anchor["_id"], {})
                candidate_attrs = compact_attrs_by_id.get(candidate["_id"], {})
                prompt_anchor_attrs = self._attrs_for_validator_prompt(anchor_attrs)
                prompt_candidate_attrs = self._attrs_for_validator_prompt(candidate_attrs)
                fast = None
                if use_fast_rules:
                    fast = self._fast_validator_decision(
                        anchor_name=anchor["nombre"],
                        candidate_name=candidate["nombre"],
                        anchor_attrs=prompt_anchor_attrs,
                        candidate_attrs=prompt_candidate_attrs,
                        similarity=float(row["similarity"]),
                        reranker_score=float(row["reranker_score"]),
                    )
                if fast is None and float(row["reranker_score"]) >= effective_th_accept:
                    if (
                        not prompt_anchor_attrs.get("brand")
                        or not prompt_candidate_attrs.get("brand")
                        or str(prompt_anchor_attrs.get("brand")).strip().lower()
                        == str(prompt_candidate_attrs.get("brand")).strip().lower()
                    ):
                        fast = {
                            "validation_score": min(0.99, 0.85 + 0.1 * float(row["reranker_score"])),
                            "review_flag": False,
                            "reason": "threshold_accept_fast_rule",
                            "decision": "aceptado",
                        }
                if fast is None and float(row["reranker_score"]) <= effective_th_reject:
                    fast = {
                        "validation_score": 0.05,
                        "review_flag": True,
                        "reason": "threshold_reject_fast_rule",
                        "decision": "rechazado",
                    }
                key = f"{anchor['_id']}|{candidate['_id']}"
                if fast is not None:
                    validation_lookup[key] = fast
                    continue

                unresolved_candidates += 1
                validation_groups.append(
                    {
                        "anchor_id": anchor["_id"],
                        "candidate_id": candidate["_id"],
                        "anchor_name": anchor["nombre"],
                        "anchor_attrs": anchor_prompt_attrs,
                        "candidate": {
                            "name": candidate.get("nombre"),
                            "attrs": prompt_candidate_attrs,
                            "url": candidate.get("url_producto"),
                            "price": candidate.get("precioFinal"),
                        },
                    }
                )

        fast_resolved_count = len(validation_lookup)

        validator_key = self._cache_key(
            {
                "stage": "validator",
                "version": "v6",
                "top_n": top_n,
                "top_k": top_k,
                "prompt": validation_prompt_id,
                "prompt_sig": validation_prompt_sig,
                "model": self.settings.vllm_validator_model_id if self.settings.use_vllm else self.settings.qwen_validator_model_id,
                "backend": "vllm" if self.settings.use_vllm else "transformers",
                "use_fast_rules": use_fast_rules,
                "groups": len(validation_groups),
                "unresolved_candidates": unresolved_candidates,
                "fast_resolved": fast_resolved_count,
            }
        )

        def load_validator_cached():
            return self._load_json_cache(session_id, "qwen_validator", validator_key)

        def compute_validator():
            return self.validator.validate_groups(
                validation_groups,
                effective_validator_batch,
                prompt_template=validation_prompt,
            )

        def save_validator_cached(payload: Any) -> None:
            self._save_json_cache(session_id, "qwen_validator", validator_key, payload)

        validations, metric = self._run_cached_stage(
            "qwen_validator",
            read_cache=use_resume and len(validation_groups) > 0,
            details={
                "groups": len(validation_groups),
                "unresolved_candidates": unresolved_candidates,
                "fast_resolved": fast_resolved_count,
                "use_fast_rules": use_fast_rules,
                "top_k": top_k,
                "batch_size": effective_validator_batch,
                "prompt": validation_prompt_id,
            },
            load_fn=load_validator_cached,
            compute_fn=compute_validator,
            save_fn=save_validator_cached if len(validation_groups) > 0 else None,
        )
        metrics.append(metric)

        for group, decision in zip(validation_groups, validations):
            anchor_id = group["anchor_id"]
            candidate_id = group["candidate_id"]
            if not isinstance(decision, dict):
                validation_lookup[f"{anchor_id}|{candidate_id}"] = {
                    "review_flag": True,
                    "reason": "sin_respuesta_modelo",
                    "cantidad": None,
                    "decision": "indeterminado",
                }
                continue

            model_decision = str(decision.get("decision", "REVIEW")).upper()
            reason_code = str(decision.get("reason_code", "AMBIGUOUS"))
            confidence = float(decision.get("confidence", 0.5))
            evidence = decision.get("evidence", [])
            reason = reason_code
            if isinstance(evidence, list) and evidence:
                reason = f"{reason_code}: {' | '.join(str(item) for item in evidence[:2])}"[:120]

            if model_decision == "ACCEPT":
                validation_lookup[f"{anchor_id}|{candidate_id}"] = {
                    "validation_score": max(0.7, confidence),
                    "review_flag": False,
                    "reason": reason,
                    "cantidad": 1,
                    "decision": "aceptado",
                }
            elif model_decision == "REJECT":
                validation_lookup[f"{anchor_id}|{candidate_id}"] = {
                    "validation_score": min(0.3, max(0.0, 1.0 - confidence)),
                    "review_flag": True,
                    "reason": reason,
                    "cantidad": None,
                    "decision": "rechazado",
                }
            else:
                validation_lookup[f"{anchor_id}|{candidate_id}"] = {
                    "validation_score": confidence,
                    "review_flag": True,
                    "reason": reason,
                    "cantidad": None,
                    "decision": "indeterminado",
                }

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
                        "razon_validacion": validation.get("reason"),
                        "cantidad_match": validation.get("cantidad"),
                        "decision_validacion": validation.get("decision"),
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

        self._attach_metric_efficiency(metrics)
        wall_seconds = time.perf_counter() - process_started
        efficiency_stats = self._build_efficiency_stats(
            metrics=metrics,
            wall_seconds=wall_seconds,
            anchors_count=len(anchors),
            products_count=len(products),
            top_n=top_n,
            top_k=top_k,
            candidate_items_count=len(candidate_items),
            unresolved_candidates=unresolved_candidates,
            fast_resolved=fast_resolved_count,
        )

        return {
            "session_id": session_id,
            "top_n": top_n,
            "top_k": top_k,
            "use_resume": use_resume,
            "extraction_prompt_id": extraction_prompt_id,
            "validation_prompt_id": validation_prompt_id,
            "use_fast_rules": use_fast_rules,
            "thresholds": {"accept": effective_th_accept, "reject": effective_th_reject},
            "results": final_results,
            "metrics": [metric.__dict__ for metric in metrics],
            "efficiency_stats": efficiency_stats,
        }
