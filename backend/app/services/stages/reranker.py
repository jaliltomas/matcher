import logging
import math
import re
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.services.stages.base import RerankerStage

logger = logging.getLogger(__name__)


def _chunks(data: list[dict[str, Any]], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def _norm(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _match_hint(left: str | None, right: str | None) -> str:
    l_norm = _norm(left)
    r_norm = _norm(right)
    if l_norm and r_norm:
        return "yes" if l_norm == r_norm else "no"
    return "unknown"


def build_pair_text(
    anchor_text: str,
    candidate_text: str,
    anchor_ner: dict[str, Any] | None,
    candidate_ner: dict[str, Any] | None,
) -> tuple[str, str]:
    anchor_ner = anchor_ner or {}
    candidate_ner = candidate_ner or {}

    anchor_brand = anchor_ner.get("brand")
    anchor_category = anchor_ner.get("category")
    candidate_brand = candidate_ner.get("brand")
    candidate_category = candidate_ner.get("category")

    brand_match = _match_hint(anchor_brand, candidate_brand)
    category_match = _match_hint(anchor_category, candidate_category)

    text_a = (
        f"ANCHOR_RAW: {anchor_text}\n"
        f"ANCHOR_BRAND: {anchor_brand or 'null'}\n"
        f"ANCHOR_CATEGORY: {anchor_category or 'null'}"
    )
    text_b = (
        f"CANDIDATE_RAW: {candidate_text}\n"
        f"CANDIDATE_BRAND: {candidate_brand or 'null'}\n"
        f"CANDIDATE_CATEGORY: {candidate_category or 'null'}\n\n"
        "HINTS:\n"
        f"BRAND_MATCH: {brand_match}\n"
        f"CATEGORY_MATCH: {category_match}"
    )
    return text_a, text_b


class XlmrRerankerStage(RerankerStage):
    def __init__(self, model_id: str, device: str, dtype: str, offload_between_stages: bool = True) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.offload_between_stages = offload_between_stages

        self.tokenizer = None
        self.model = None

    def _torch_dtype(self) -> torch.dtype:
        if self.device.startswith("cuda") and self.dtype == "float16":
            return torch.float16
        return torch.float32

    def _load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        logger.info("Cargando reranker XLM-R: %s", self.model_id)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        except Exception:
            logger.warning("No se pudo cargar tokenizer fast para %s, usando slow tokenizer.", self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            torch_dtype=self._torch_dtype(),
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def _unload(self) -> None:
        if not self.offload_between_stages:
            return

        self.tokenizer = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _entailment_index(self) -> int:
        assert self.model is not None
        id2label = getattr(self.model.config, "id2label", {}) or {}
        for idx, label in id2label.items():
            if "entail" in str(label).lower():
                return int(idx)
        return max(id2label.keys()) if id2label else 0

    # Etapa 5 del pipeline: reranking cross-encoder para reordenar candidatos.
    def score_pairs(self, pairs: list[dict[str, Any]], batch_size: int) -> list[float]:
        if not pairs:
            return []

        self._load()
        assert self.model is not None
        assert self.tokenizer is not None

        entailment_idx = self._entailment_index()
        scores: list[float] = []
        total_batches = math.ceil(len(pairs) / batch_size)

        for batch in tqdm(_chunks(pairs, batch_size), total=total_batches, desc="Reranker", leave=False):
            left = [item.get("text_a") or item["anchor_text"] for item in batch]
            right = [item.get("text_b") or item["candidate_text"] for item in batch]

            encoded = self.tokenizer(
                left,
                right,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(self.device)

            with torch.inference_mode():
                logits = self.model(**encoded).logits.float()

            if logits.shape[-1] == 1:
                batch_scores = torch.sigmoid(logits.squeeze(-1)).tolist()
            else:
                probs = torch.softmax(logits, dim=-1)
                batch_scores = probs[:, entailment_idx].tolist()
            scores.extend([float(value) for value in batch_scores])

        self._unload()
        return scores
