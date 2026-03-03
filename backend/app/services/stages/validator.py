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

_ALLOWED_DECISIONS = {"ACCEPT", "REJECT", "REVIEW"}
_ALLOWED_REASON_CODES = {
    "SAME_PRODUCT",
    "BRAND_MISMATCH",
    "CATEGORY_MISMATCH",
    "INSUFFICIENT_EVIDENCE",
    "TOO_GENERIC",
    "AMBIGUOUS",
}


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
        candidate = group["candidate"]
        payload = {
            "anchor": {
                "name": group["anchor_name"],
                "attrs": group.get("anchor_attrs", {}),
            },
            "candidate": {
                "name": candidate.get("name"),
                "attrs": candidate.get("attrs", {}),
                "url": candidate.get("url"),
                "price": candidate.get("price"),
            },
        }
        if prompt_template:
            prompt = prompt_template
            prompt = prompt.replace("{ANCHOR_NAME}", str(group["anchor_name"]))
            prompt = prompt.replace("{ANCHOR_ATTRS}", json.dumps(group["anchor_attrs"], ensure_ascii=False))
            prompt = prompt.replace("{CANDIDATES_JSON}", json.dumps([candidate], ensure_ascii=False))
            prompt = prompt.replace("{PAIR_JSON}", json.dumps(payload, ensure_ascii=False))
            return prompt

        return (
            "Eres un auditor de matching de productos. Evalua SOLO un par anchor-candidate. "
            "Responde SOLO JSON valido con esquema exacto: "
            '{"decision":"ACCEPT|REJECT|REVIEW","reason_code":"SAME_PRODUCT|BRAND_MISMATCH|CATEGORY_MISMATCH|INSUFFICIENT_EVIDENCE|TOO_GENERIC|AMBIGUOUS","confidence":0.0,"evidence":[]}. '
            "Reglas duras: (1) si ambas brands existen y difieren => REJECT con confidence >= 0.9 y reason_code BRAND_MISMATCH. "
            "(2) Si category existe en ambos y difiere claramente => REJECT con CATEGORY_MISMATCH. "
            "(3) Si no puedes citar evidencia textual exacta que soporte ACCEPT => usa REVIEW. "
            "(4) evidence debe contener hasta 3 substrings exactos del texto de entrada. "
            "(5) Nunca inventes brand/category faltantes.\n"
            f"PAIR_JSON: {json.dumps(payload, ensure_ascii=False)}"
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
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            eos_ids.append(eos_token_id)
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0:
            eos_ids.append(im_end_id)
        if not eos_ids:
            return 0
        return eos_ids[0] if len(eos_ids) == 1 else eos_ids

    def _clean_evidence(self, pair_text: str, evidence: Any) -> list[str]:
        if not isinstance(evidence, list):
            return []
        out: list[str] = []
        for item in evidence:
            quote = str(item).strip()
            if not quote:
                continue
            if quote in pair_text:
                out.append(quote[:120])
            if len(out) >= 3:
                break
        return out

    def _parse_auditor_json(self, output_text: str, pair_text: str) -> dict[str, Any]:
        parsed = extract_first_json_object(output_text)
        fallback = {
            "decision": "REVIEW",
            "reason_code": "INSUFFICIENT_EVIDENCE",
            "confidence": 0.5,
            "evidence": [],
        }
        if not isinstance(parsed, dict):
            return fallback

        decision = str(parsed.get("decision", "REVIEW")).upper()
        reason_code = str(parsed.get("reason_code", "INSUFFICIENT_EVIDENCE")).upper()
        try:
            confidence = float(parsed.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        evidence = self._clean_evidence(pair_text, parsed.get("evidence", []))

        if decision not in _ALLOWED_DECISIONS:
            decision = "REVIEW"
        if reason_code not in _ALLOWED_REASON_CODES:
            reason_code = "AMBIGUOUS"
        if decision == "ACCEPT" and not evidence:
            decision = "REVIEW"
            reason_code = "INSUFFICIENT_EVIDENCE"

        return {
            "decision": decision,
            "reason_code": reason_code,
            "confidence": confidence,
            "evidence": evidence,
        }

    def _hard_rule_override(self, group: dict[str, Any]) -> dict[str, Any] | None:
        anchor_attrs = group.get("anchor_attrs", {}) if isinstance(group.get("anchor_attrs"), dict) else {}
        candidate = group.get("candidate", {}) if isinstance(group.get("candidate"), dict) else {}
        candidate_attrs = candidate.get("attrs", {}) if isinstance(candidate.get("attrs"), dict) else {}

        a_brand = str(anchor_attrs.get("brand") or "").strip().lower()
        c_brand = str(candidate_attrs.get("brand") or "").strip().lower()
        if a_brand and c_brand and a_brand != c_brand:
            return {
                "decision": "REJECT",
                "reason_code": "BRAND_MISMATCH",
                "confidence": 0.95,
                "evidence": [a_brand[:80], c_brand[:80]],
            }

        a_cat = str(anchor_attrs.get("category") or "").strip().lower()
        c_cat = str(candidate_attrs.get("category") or "").strip().lower()
        if a_cat and c_cat and a_cat != c_cat:
            return {
                "decision": "REJECT",
                "reason_code": "CATEGORY_MISMATCH",
                "confidence": 0.92,
                "evidence": [a_cat[:80], c_cat[:80]],
            }
        return None

    def _max_new_tokens_for_batch(self, batch: list[dict[str, Any]]) -> int:
        _ = batch
        return int(min(384, max(128, self.max_new_tokens)))

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
            max_tokens = max(128, self._max_new_tokens_for_batch(batch))
            try:
                decoded = self.vllm_client.complete_many(
                    prompts,
                    max_tokens=max_tokens,
                    workers=min(batch_size, 4),
                    temperature=0.0,
                    top_p=1.0,
                )
            except Exception as exc:
                logger.warning("Qwen Validator(vLLM) fallo: %s", exc)
                raise
            for text, group in zip(decoded, batch):
                pair_text = f"{group.get('anchor_name','')} || {group.get('candidate',{}).get('name','')}"
                parsed = self._parse_auditor_json(text, pair_text)
                override = self._hard_rule_override(group)
                outputs.append(override or parsed)
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
                    top_p=1.0,
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self._eos_token_ids(),
                )

            prompt_length = encoded["input_ids"].shape[1]
            decoded = self.tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)
            for text, group in zip(decoded, batch):
                pair_text = f"{group.get('anchor_name','')} || {group.get('candidate',{}).get('name','')}"
                parsed = self._parse_auditor_json(text, pair_text)
                override = self._hard_rule_override(group)
                outputs.append(override or parsed)

        self._unload()
        return outputs
