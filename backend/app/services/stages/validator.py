import json
import logging
import math
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        offload_between_stages: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.offload_between_stages = offload_between_stages

        self.tokenizer: Any = None
        self.model: Any = None

    def _torch_dtype(self) -> torch.dtype:
        if self.device.startswith("cuda") and self.dtype == "float16":
            return torch.float16
        return torch.float32

    def _load(self) -> None:
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
        if not self.offload_between_stages:
            return

        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prompt(self, pair: dict[str, Any], prompt_template: str | None) -> str:
        if prompt_template:
            prompt = prompt_template
            prompt = prompt.replace("{ANCHOR_NAME}", str(pair["anchor_name"]))
            prompt = prompt.replace("{ANCHOR_ATTRS}", json.dumps(pair["anchor_attrs"], ensure_ascii=False))
            prompt = prompt.replace("{CANDIDATE_NAME}", str(pair["candidate_name"]))
            prompt = prompt.replace("{CANDIDATE_ATTRS}", json.dumps(pair["candidate_attrs"], ensure_ascii=False))
            prompt = prompt.replace("{URL}", str(pair.get("url", "")))
            prompt = prompt.replace("{PRICE}", str(pair.get("price", None)))
            return prompt

        return (
            "Evalua si ANCLA y CANDIDATO son el mismo SKU. "
            "Responde SOLO JSON valido: "
            '{"validation_score":0.0,"review_flag":false,"reason":""}. '
            "score entre 0 y 1, sin texto adicional.\n"
            f"ANCLA: {pair['anchor_name']}\n"
            f"ATRIBUTOS_ANCLA: {json.dumps(pair['anchor_attrs'], ensure_ascii=False)}\n"
            f"CANDIDATO: {pair['candidate_name']}\n"
            f"ATRIBUTOS_CANDIDATO: {json.dumps(pair['candidate_attrs'], ensure_ascii=False)}\n"
            f"URL: {pair.get('url', '')}\n"
            f"PRECIO: {pair.get('price', None)}"
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

    def _parse_json(self, output_text: str) -> dict[str, Any]:
        parsed = extract_first_json_object(output_text)
        try:
            if parsed is None:
                raise ValueError("No JSON parsed")
            score = float(parsed.get("validation_score", 0.0))
            return {
                "validation_score": max(0.0, min(1.0, score)),
                "review_flag": bool(parsed.get("review_flag", score < 0.65)),
                "reason": str(parsed.get("reason", ""))[:200],
            }
        except Exception:
            return {
                "validation_score": 0.0,
                "review_flag": True,
                "reason": output_text[:200],
            }

    # Etapa 6 del pipeline: validacion final con Qwen-7B para score de confianza.
    def validate_pairs(
        self,
        pairs: list[dict[str, Any]],
        batch_size: int,
        prompt_template: str | None = None,
    ) -> list[dict[str, Any]]:
        if not pairs:
            return []

        self._load()
        assert self.model is not None
        assert self.tokenizer is not None

        outputs: list[dict[str, Any]] = []
        total_batches = math.ceil(len(pairs) / batch_size)
        for batch in tqdm(_chunks(pairs, batch_size), total=total_batches, desc="Validator", leave=False):
            prompts = [self._prompt(pair, prompt_template) for pair in batch]
            encoded = self._build_batch_inputs(prompts).to(self.model.device)

            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=min(self.max_new_tokens, 56),
                    do_sample=False,
                    temperature=0.0,
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self._eos_token_ids(),
                )

            prompt_length = encoded["input_ids"].shape[1]
            decoded = self.tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)
            outputs.extend([self._parse_json(text) for text in decoded])

        self._unload()
        return outputs
