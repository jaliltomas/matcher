import json
import logging
import math
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.services.vllm_client import VllmChatClient
from app.services.stages.json_parsing import extract_first_json_object
from app.services.stages.base import ValidatorStage

logger = logging.getLogger(__name__)


def _chunks(data: list[dict[str, Any]], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


class QwenValidatorStage(ValidatorStage):
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
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.use_vllm = use_vllm
        self.offload_between_stages = offload_between_stages

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
        logger.warning("Qwen Validator: fallback de vLLM a transformers local")
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

        logger.info("Cargando Qwen Validator: %s", self.model_id)
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

    def _prompt(self, group: dict[str, Any], prompt_template: str | None) -> str:
        candidates_json = json.dumps(group["candidates"], ensure_ascii=False)
        if prompt_template:
            prompt = prompt_template
            prompt = prompt.replace("{ANCHOR_NAME}", str(group["anchor_name"]))
            prompt = prompt.replace("{ANCHOR_ATTRS}", json.dumps(group["anchor_attrs"], ensure_ascii=False))
            prompt = prompt.replace("{CANDIDATES_JSON}", candidates_json)
            return prompt

        return (
            "Evalua candidatos contra un ancla y decide si son el mismo SKU. "
            "Responde SOLO JSON valido (sin texto extra) con formato exacto:\n"
            '{"matches_validos":[{"id":1,"razon":"","cantidad":1}],"rechazados":[{"id":2,"razon":""}]}.\n'
            "Todos los candidatos deben quedar en una de las listas.\n"
            f"ANCLA: {group['anchor_name']}\n"
            f"ATRIBUTOS_ANCLA: {json.dumps(group['anchor_attrs'], ensure_ascii=False)}\n"
            f"CANDIDATOS: {candidates_json}"
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

    def _parse_json(self, output_text: str, allowed_ids: list[int]) -> dict[str, Any]:
        parsed = extract_first_json_object(output_text)

        allowed = {int(value) for value in allowed_ids}
        matches_validos: list[dict[str, Any]] = []
        rechazados: list[dict[str, Any]] = []

        if isinstance(parsed, dict):
            raw_matches = parsed.get("matches_validos", [])
            raw_rejects = parsed.get("rechazados", [])

            if isinstance(raw_matches, list):
                for row in raw_matches:
                    if not isinstance(row, dict):
                        continue
                    try:
                        idx = int(row.get("id"))
                    except Exception:
                        continue
                    if idx not in allowed:
                        continue
                    reason = str(row.get("razon", "match"))[:80]
                    try:
                        quantity = int(row.get("cantidad", 1))
                    except Exception:
                        quantity = 1
                    matches_validos.append({"id": idx, "razon": reason, "cantidad": max(1, quantity)})

            if isinstance(raw_rejects, list):
                for row in raw_rejects:
                    if not isinstance(row, dict):
                        continue
                    try:
                        idx = int(row.get("id"))
                    except Exception:
                        continue
                    if idx not in allowed:
                        continue
                    reason = str(row.get("razon", "rechazado"))[:100]
                    rechazados.append({"id": idx, "razon": reason})

        valid_ids = {item["id"] for item in matches_validos}
        reject_ids = {item["id"] for item in rechazados}
        missing = [idx for idx in allowed_ids if idx not in valid_ids and idx not in reject_ids]
        sin_respuesta = [{"id": int(idx), "razon": "sin_respuesta_modelo"} for idx in missing]

        return {"matches_validos": matches_validos, "rechazados": rechazados, "sin_respuesta": sin_respuesta}

    def _max_new_tokens_for_batch(self, batch: list[dict[str, Any]]) -> int:
        estimated = 0
        for group in batch:
            candidates = len(group.get("candidates", []))
            estimated = max(estimated, 48 + (24 * candidates))
        return int(min(384, max(self.max_new_tokens, estimated)))

    def _validate_with_vllm(
        self,
        groups: list[dict[str, Any]],
        batch_size: int,
        prompt_template: str | None,
    ) -> list[dict[str, Any]]:
        assert self.vllm_client is not None
        outputs: list[dict[str, Any]] = []
        total_batches = math.ceil(len(groups) / batch_size)
        for batch in tqdm(_chunks(groups, batch_size), total=total_batches, desc="Validator(vLLM)", leave=False):
            prompts = [self._prompt(group, prompt_template) for group in batch]
            max_tokens = self._max_new_tokens_for_batch(batch)
            try:
                decoded = self.vllm_client.complete_many(
                    prompts,
                    max_tokens=max_tokens,
                    workers=min(batch_size, 4),
                )
            except Exception as exc:
                logger.warning("Qwen Validator(vLLM) fallo: %s", exc)
                raise
            for text, group in zip(decoded, batch):
                allowed_ids = [int(item.get("id")) for item in group.get("candidates", []) if "id" in item]
                outputs.append(self._parse_json(text, allowed_ids))
        return outputs

    # Etapa 6 del pipeline: validacion final por ancla devolviendo aceptados y rechazados.
    def validate_groups(
        self,
        groups: list[dict[str, Any]],
        batch_size: int,
        prompt_template: str | None = None,
    ) -> list[dict[str, Any]]:
        if not groups:
            return []

        if self.use_vllm:
            try:
                return self._validate_with_vllm(groups, batch_size, prompt_template)
            except Exception:
                self._switch_to_local_fallback()

        self._load()
        assert self.model is not None
        assert self.tokenizer is not None

        outputs: list[dict[str, Any]] = []
        total_batches = math.ceil(len(groups) / batch_size)
        for batch in tqdm(_chunks(groups, batch_size), total=total_batches, desc="Validator", leave=False):
            prompts = [self._prompt(group, prompt_template) for group in batch]
            encoded = self._build_batch_inputs(prompts).to(self.model.device)
            max_tokens = self._max_new_tokens_for_batch(batch)

            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=0.0,
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self._eos_token_ids(),
                )

            prompt_length = encoded["input_ids"].shape[1]
            decoded = self.tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)
            for text, group in zip(decoded, batch):
                allowed_ids = [int(item.get("id")) for item in group.get("candidates", []) if "id" in item]
                outputs.append(self._parse_json(text, allowed_ids))

        self._unload()
        return outputs
