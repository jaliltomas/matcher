import logging
import math
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.services.stages.base import RerankerStage

logger = logging.getLogger(__name__)


def _chunks(data: list[dict[str, Any]], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


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
            left = [item["anchor_text"] for item in batch]
            right = [item["candidate_text"] for item in batch]

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
