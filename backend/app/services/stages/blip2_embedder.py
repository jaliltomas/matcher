import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Model, Blip2Processor

from app.services.image_utils import load_image
from app.services.stages.base import EmbeddingStage

logger = logging.getLogger(__name__)


def _chunks(data: list[dict[str, Any]], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


class Blip2EmbeddingStage(EmbeddingStage):
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str = "float16",
        offload_between_stages: bool = True,
        image_download_workers: int = 24,
        image_timeout_seconds: int = 4,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.offload_between_stages = offload_between_stages
        self.image_download_workers = image_download_workers
        self.image_timeout_seconds = image_timeout_seconds

        self.processor: Any = None
        self.model: Any = None
        self._placeholder = self._placeholder_image()
        self._image_cache: dict[str, Image.Image] = {}
        self._image_cache_lock = threading.Lock()

    def _torch_dtype(self) -> torch.dtype:
        if self.device.startswith("cuda") and self.dtype == "float16":
            return torch.float16
        return torch.float32

    def _load(self) -> None:
        if self.processor is not None and self.model is not None:
            return

        logger.info("Cargando BLIP-2: %s", self.model_id)
        self.processor = Blip2Processor.from_pretrained(self.model_id)
        self.model = Blip2Model.from_pretrained(
            self.model_id,
            torch_dtype=self._torch_dtype(),
            low_cpu_mem_usage=True,
        )
        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        self.model.to(self.device)
        self.model.eval()

    def _unload(self) -> None:
        if not self.offload_between_stages:
            return

        self.processor = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _compose_text(self, record: dict[str, Any]) -> str:
        chunks = [
            record.get("nombre", ""),
            record.get("sitio", ""),
            record.get("seller", ""),
            record.get("textoffer", ""),
        ]
        return " | ".join([part.strip() for part in chunks if part and part.strip()])

    def _placeholder_image(self) -> Image.Image:
        return Image.new("RGB", (224, 224), color=(245, 245, 245))

    def _load_one_image(self, image_value: str | None) -> Image.Image:
        if not image_value:
            return self._placeholder

        with self._image_cache_lock:
            cached = self._image_cache.get(image_value)
        if cached is not None:
            return cached

        image = load_image(image_value, timeout=self.image_timeout_seconds) or self._placeholder
        with self._image_cache_lock:
            if len(self._image_cache) > 4000:
                self._image_cache.clear()
            self._image_cache[image_value] = image
        return image

    def _load_images_parallel_with_pool(
        self,
        batch: list[dict[str, Any]],
        image_pool: ThreadPoolExecutor,
    ) -> list[Image.Image]:
        return list(image_pool.map(lambda item: self._load_one_image(item.get("img")), batch))

    def _prepare_io_batch(
        self,
        batch: list[dict[str, Any]],
        image_pool: ThreadPoolExecutor,
    ) -> tuple[list[str], list[Image.Image]]:
        texts = [self._compose_text(item) for item in batch]
        images = self._load_images_parallel_with_pool(batch, image_pool)
        return texts, images

    # Etapa 1 del pipeline: embedding multimodal con BLIP-2 (texto + imagen).
    def embed_records(self, records: list[dict[str, Any]], batch_size: int) -> np.ndarray:
        if not records:
            return np.zeros((0, 768), dtype=np.float32)

        self._load()
        assert self.processor is not None
        assert self.model is not None

        logger.info(
            "BLIP2 embedding records=%d batch=%d workers=%d device=%s",
            len(records),
            batch_size,
            self.image_download_workers,
            self.device,
        )

        vectors: list[np.ndarray] = []
        started = perf_counter()
        total_batches = math.ceil(len(records) / batch_size)
        batch_iter = iter(_chunks(records, batch_size))

        image_workers = max(1, min(self.image_download_workers, batch_size))
        with ThreadPoolExecutor(max_workers=image_workers) as image_pool, ThreadPoolExecutor(max_workers=1) as prefetch_pool:
            current_batch = next(batch_iter, None)
            current_prefetch = (
                prefetch_pool.submit(self._prepare_io_batch, current_batch, image_pool)
                if current_batch is not None
                else None
            )

            for _ in tqdm(range(total_batches), total=total_batches, desc="BLIP2 embeddings", leave=False):
                if current_batch is None or current_prefetch is None:
                    break

                texts, images = current_prefetch.result()
                next_batch = next(batch_iter, None)
                next_prefetch = (
                    prefetch_pool.submit(self._prepare_io_batch, next_batch, image_pool)
                    if next_batch is not None
                    else None
                )

                inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

                with torch.inference_mode():
                    if self.device.startswith("cuda") and self.dtype == "float16":
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = self.model(**inputs, return_dict=True)
                    else:
                        outputs = self.model(**inputs, return_dict=True)
                    if hasattr(outputs, "qformer_outputs") and outputs.qformer_outputs is not None:
                        hidden = outputs.qformer_outputs.last_hidden_state
                        pooled = hidden.mean(dim=1)
                    elif hasattr(outputs, "last_hidden_state"):
                        pooled = outputs.last_hidden_state.mean(dim=1)
                    else:
                        pooled = outputs.pooler_output

                    pooled = F.normalize(pooled.float(), p=2, dim=1)

                vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
                current_batch = next_batch
                current_prefetch = next_prefetch

        self._unload()
        elapsed = max(1e-6, perf_counter() - started)
        logger.info("BLIP2 throughput %.2f items/s", len(records) / elapsed)
        return np.concatenate(vectors, axis=0)
