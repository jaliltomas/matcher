import json
import logging
import math
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.services.vllm_client import VllmChatClient
from app.services.stages.json_parsing import extract_first_json_object, normalize_attributes
from app.services.stages.base import EnricherStage

logger = logging.getLogger(__name__)


def _chunks(data: list[dict[str, Any]], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


class QwenNerEnricherStage(EnricherStage):
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str,
        max_new_tokens: int,
        use_vllm: bool,
        vllm_base_url: str,
        vllm_api_key: str,
        vllm_timeout_seconds: int,
        vllm_max_retries: int,
        vllm_disable_thinking: bool,
        vllm_max_parallel: int,
        offload_between_stages: bool = True,
        strict_json: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.use_vllm = use_vllm
        self.offload_between_stages = offload_between_stages
        self.strict_json = strict_json
        self.parse_fail_count = 0
        self.total_count = 0

        self.tokenizer: Any = None
        self.model: Any = None
        self.vllm_client: VllmChatClient | None = None
        if self.use_vllm:
            self.vllm_client = VllmChatClient(
                base_url=vllm_base_url,
                model_id=model_id,
                api_key=vllm_api_key,
                timeout_seconds=vllm_timeout_seconds,
                max_retries=vllm_max_retries,
                disable_thinking=vllm_disable_thinking,
                max_parallel=vllm_max_parallel,
            )

    def _switch_to_local_fallback(self) -> None:
        if not self.use_vllm:
            return
        logger.warning("Qwen NER: fallback de vLLM a transformers local")
        self.use_vllm = False
        self.vllm_client = None

    def _torch_dtype(self) -> torch.dtype:
        if self.device.startswith("cuda") and self.dtype == "float16":
            return torch.float16
        return torch.float32

    def _load(self) -> None:
        if self.use_vllm:
            return
        if self.model is not None and self.tokenizer is not None:
            return

        logger.info("Cargando Qwen NER: %s", self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self._torch_dtype(),
            device_map="auto" if self.device.startswith("cuda") else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def _unload(self) -> None:
        if self.use_vllm:
            return
        if not self.offload_between_stages:
            return

        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prompt(self, text: str, prompt_template: str | None) -> str:
        if prompt_template:
            return prompt_template.replace("{TEXT}", text)

        return (
            "Extrae SOLO brand y category del texto de producto. "
            "Responde SOLO JSON valido (sin markdown) con esquema exacto: "
            '{"brand":null,"category":null,"evidence":{"brand":null,"category":null}}. '
            "Reglas: (1) No adivinar. (2) Si no hay evidencia literal, usa null. "
            "(3) evidence.brand y evidence.category deben ser substrings exactos del texto. "
            "(4) No confundas seller/tienda con brand. Texto: "
            f"{text}"
        )

    def _build_batch_inputs(self, prompts: list[str]):
        assert self.tokenizer is not None
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                [
                    {"role": "system", "content": "Devuelve solo JSON valido sin explicaciones."},
                    {"role": "user", "content": prompt},
                ]
                for prompt in prompts
            ]
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return self.tokenizer(rendered, return_tensors="pt", padding=True, truncation=True)
        return self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    def _eos_token_ids(self) -> list[int] | int:
        assert self.tokenizer is not None
        eos_ids: list[int] = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(int(self.tokenizer.eos_token_id))
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0:
            eos_ids.append(im_end_id)
        if not eos_ids:
            return 0
        return eos_ids[0] if len(eos_ids) == 1 else eos_ids

    def _empty_payload(self, raw_fallback: str | None = None) -> dict[str, Any]:
        _ = raw_fallback
        return {
            "brand": None,
            "category": None,
            "evidence": {"brand": None, "category": None},
        }

    def _normalize_evidence(self, source_text: str, value: str | None) -> str | None:
        if not value:
            return None
        quote = str(value).strip()
        if not quote:
            return None
        return quote if quote in source_text else None

    def _parse_json(self, source_text: str, output_text: str) -> dict[str, Any]:
        parsed = extract_first_json_object(output_text)
        if parsed is None:
            self.parse_fail_count += 1
            logger.warning("Qwen NER parse_fail: no JSON object")
            return self._empty_payload(raw_fallback=output_text) if self.strict_json else normalize_attributes({}, raw_fallback=output_text)

        normalized = normalize_attributes(parsed, raw_fallback=output_text)
        evidence = normalized.get("evidence", {}) if isinstance(normalized.get("evidence"), dict) else {}
        normalized["evidence"] = {
            "brand": self._normalize_evidence(source_text, evidence.get("brand")),
            "category": self._normalize_evidence(source_text, evidence.get("category")),
        }
        if self.strict_json:
            normalized = {
                "brand": normalized.get("brand"),
                "category": normalized.get("category"),
                "evidence": normalized.get("evidence", {"brand": None, "category": None}),
            }
        return normalized

    def _extract_with_vllm(
        self,
        items: list[dict[str, Any]],
        batch_size: int,
        prompt_template: str | None,
    ) -> dict[str, dict[str, Any]]:
        assert self.vllm_client is not None
        result: dict[str, dict[str, Any]] = {}
        total_batches = math.ceil(len(items) / batch_size)
        for batch in tqdm(_chunks(items, batch_size), total=total_batches, desc="Qwen NER(vLLM)", leave=False):
            prompts = [self._prompt(item.get("nombre", ""), prompt_template) for item in batch]
            try:
                outputs = self.vllm_client.complete_many(
                    prompts,
                    max_tokens=min(self.max_new_tokens, 256),
                    workers=min(batch_size, 8),
                    temperature=0.0,
                    top_p=1.0,
                )
            except Exception as exc:
                logger.warning("Qwen NER(vLLM) fallo: %s", exc)
                raise
            for item, output in zip(batch, outputs):
                result[item["_id"]] = self._parse_json(item.get("nombre", ""), output)
        return result

    # Etapa 4 del pipeline: enriquecimiento semantico con Qwen-7B para NER.
    def extract_attributes(
        self,
        items: list[dict[str, Any]],
        batch_size: int,
        prompt_template: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        if not items:
            return {}
        self.parse_fail_count = 0
        self.total_count = len(items)

        if self.use_vllm:
            try:
                return self._extract_with_vllm(items, batch_size, prompt_template)
            except Exception:
                self._switch_to_local_fallback()

        self._load()
        assert self.model is not None
        assert self.tokenizer is not None

        result: dict[str, dict[str, Any]] = {}
        total_batches = math.ceil(len(items) / batch_size)
        for batch in tqdm(_chunks(items, batch_size), total=total_batches, desc="Qwen NER", leave=False):
            prompts = [self._prompt(item.get("nombre", ""), prompt_template) for item in batch]
            encoded = self._build_batch_inputs(prompts).to(self.model.device)

            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=min(self.max_new_tokens, 256),
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self._eos_token_ids(),
                )

            prompt_length = encoded["input_ids"].shape[1]
            decoded = self.tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)

            for item, output in zip(batch, decoded):
                result[item["_id"]] = self._parse_json(item.get("nombre", ""), output)

        self._unload()
        return result
