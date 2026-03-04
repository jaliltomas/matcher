import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal

from app.core.config import get_settings
from app.services.stages.json_parsing import extract_first_json_object
from app.services.vllm_client import VllmChatClient

logger = logging.getLogger(__name__)

PromptKind = Literal["extraction", "validation"]

_META_PROMPT_TEMPLATE = """
You are a senior ML prompt engineer designing production prompts for an e-commerce product matching system.

You must output ONLY valid JSON with this schema:
{
  "ner_prompt": string,
  "validator_prompt": string,
  "meta": {
    "assumptions": string[],
    "recommended_thresholds": {"accept": number, "reject": number},
    "notes": string[]
  }
}

Context:
We have a matching platform. For each anchor and candidate product we have text (and possibly other metadata).
We run:
1) NER-like extraction on each text to extract ONLY:
- brand (string or null)
- category (string or null)
- evidence quotes (exact substrings) supporting brand/category.
2) A final validator that decides if anchor and candidate are the SAME PRODUCT.

Your job: write two robust prompts (NER + Validator) tailored to this business vertical.

Requirements for the NER prompt:
- Must enforce JSON-only output with strict schema:
  {
    "brand": string|null,
    "category": string|null,
    "evidence": {"brand": string|null, "category": string|null}
  }
- Must forbid guessing / inference. If not explicitly supported by text => null.
- Evidence must be exact substrings copied from the input.
- Must include rules to avoid confusing seller/store with brand.
- Must be deterministic-friendly (instruct temp=0 usage; concise).

Requirements for the Validator prompt:
- Must enforce JSON-only output with strict schema:
  {
    "decision": "ACCEPT"|"REJECT"|"REVIEW",
    "reason_code": "SAME_PRODUCT"|"BRAND_MISMATCH"|"CATEGORY_MISMATCH"|"INSUFFICIENT_EVIDENCE"|"TOO_GENERIC"|"AMBIGUOUS",
    "confidence": number,
    "evidence": string[]
  }
- Hard rules:
  - If both brands are present and differ => REJECT with high confidence.
  - If cannot quote evidence supporting match => REVIEW, not ACCEPT.
  - Do not invent missing brand/category; treat as unknown.
- Must include vertical-specific guidance (what counts as same product vs variant) based on provided edge cases.

Inputs you must incorporate:
- Language: {{language}}
- Vertical description: {{vertical_description}}
- Brand notes: {{brand_notes}}
- Category taxonomy (optional; if provided, category must be one of these or null): {{category_taxonomy}}
- Edge cases: {{edge_cases}}

Also provide:
- recommended_thresholds:
  - accept: suggest a reranker score threshold above which we can auto-ACCEPT when brand matches.
  - reject: suggest a reranker score threshold below which we can auto-REJECT.
Use conservative defaults if uncertain (e.g., accept 0.80, reject 0.35).
Add notes about what to tune per client.

Return ONLY the JSON object. No other text.
""".strip()


EXTRACTION_PRESETS = [
    {
        "id": "default_ner_v2",
        "name": "Default Brand+Category",
        "description": "Extraccion deterministica de brand/category con evidencia textual.",
        "template": (
            "Extrae SOLO brand y category del texto de producto. "
            "Responde SOLO JSON valido con esquema exacto: "
            '{"brand":null,"category":null,"evidence":{"brand":null,"category":null}}. '
            "Reglas: no adivinar; evidencia debe ser substring exacto del texto; si falta evidencia usar null; "
            "no confundas seller/tienda con brand; salida sin markdown ni texto extra. Texto: {TEXT}"
        ),
    }
]


VALIDATION_PRESETS = [
    {
        "id": "default_validator_auditor_v2",
        "name": "Auditor Mode",
        "description": "Validator deterministico con decision, reason_code, confidence y evidencia.",
        "template": (
            "Eres auditor de matching de productos. Evalua SOLO un par. "
            "Responde SOLO JSON valido con esquema exacto: "
            '{"decision":"ACCEPT|REJECT|REVIEW","reason_code":"SAME_PRODUCT|BRAND_MISMATCH|CATEGORY_MISMATCH|INSUFFICIENT_EVIDENCE|TOO_GENERIC|AMBIGUOUS","confidence":0.0,"evidence":[]}. '
            "Reglas duras: brand mismatch => REJECT >=0.9; category mismatch claro => REJECT; "
            "si no puedes citar evidencia textual exacta para aceptar => REVIEW; no inventar brand/category faltantes; "
            "evidence max 2 substrings exactos. "
            "No incluyas cadena de pensamiento ni explicaciones; solo el JSON final. "
            "Pair: {PAIR_JSON}"
        ),
    }
]


def _presets_dir() -> Path:
    settings = get_settings()
    path = settings.data_dir / "prompt_presets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_generated_presets() -> dict[str, list[dict[str, str]]]:
    extraction: list[dict[str, str]] = []
    validation: list[dict[str, str]] = []
    for path in sorted(_presets_dir().glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        pid = str(payload.get("id", "")).strip()
        if not pid:
            continue
        ner_prompt = str(payload.get("ner_prompt", "")).strip()
        validator_prompt = str(payload.get("validator_prompt", "")).strip()
        if ner_prompt:
            extraction.append(
                {
                    "id": f"{pid}_ner",
                    "name": f"Generated {pid} NER",
                    "description": "Prompt generado por /prompts/generate",
                    "template": ner_prompt,
                }
            )
        if validator_prompt:
            validation.append(
                {
                    "id": f"{pid}_validator",
                    "name": f"Generated {pid} Validator",
                    "description": "Prompt generado por /prompts/generate",
                    "template": validator_prompt,
                }
            )
    return {"extraction": extraction, "validation": validation}


def list_prompt_presets() -> dict:
    generated = _load_generated_presets()
    extraction = EXTRACTION_PRESETS + generated["extraction"]
    validation = VALIDATION_PRESETS + generated["validation"]
    return {
        "extraction": extraction,
        "validation": validation,
        "defaults": {
            "extraction": EXTRACTION_PRESETS[0]["id"],
            "validation": VALIDATION_PRESETS[0]["id"],
        },
    }


def _pick_preset(kind: PromptKind, preset_id: str | None) -> dict:
    presets = list_prompt_presets()["extraction" if kind == "extraction" else "validation"]
    if preset_id:
        for preset in presets:
            if preset["id"] == preset_id:
                return preset
    return presets[0]


def resolve_prompt(
    kind: PromptKind,
    preset_id: str | None,
    custom_prompt: str | None,
) -> tuple[str, str]:
    if custom_prompt and custom_prompt.strip():
        clean = custom_prompt.strip()
        digest = hashlib.sha1(clean.encode("utf-8")).hexdigest()[:10]
        return clean, f"custom_{digest}"

    preset = _pick_preset(kind, preset_id)
    return str(preset["template"]), str(preset["id"])


def _build_meta_prompt(payload: dict[str, Any], strict_suffix: str = "") -> str:
    prompt = _META_PROMPT_TEMPLATE
    prompt = prompt.replace("{{language}}", str(payload.get("language", "es")))
    prompt = prompt.replace("{{vertical_description}}", str(payload.get("vertical_description", "")))
    prompt = prompt.replace("{{brand_notes}}", str(payload.get("brand_notes") or "null"))
    prompt = prompt.replace("{{category_taxonomy}}", json.dumps(payload.get("category_taxonomy"), ensure_ascii=False))
    prompt = prompt.replace("{{edge_cases}}", json.dumps(payload.get("edge_cases"), ensure_ascii=False))
    if strict_suffix:
        prompt = f"{prompt}\n\n{strict_suffix}"
    return prompt


def _validate_generated_payload(parsed: dict[str, Any]) -> bool:
    if not isinstance(parsed, dict):
        return False
    ner_prompt = parsed.get("ner_prompt")
    validator_prompt = parsed.get("validator_prompt")
    meta = parsed.get("meta")
    if not isinstance(ner_prompt, str) or not isinstance(validator_prompt, str) or not isinstance(meta, dict):
        return False
    ner_ok = all(token in ner_prompt for token in ["brand", "category", "evidence"])
    validator_ok = all(token in validator_prompt for token in ["decision", "reason_code", "confidence", "evidence"])
    return ner_ok and validator_ok


def _save_generated_prompt_bundle(payload: dict[str, Any]) -> str:
    signature = hashlib.sha1(
        json.dumps(
            {
                "vertical_description": payload.get("vertical_description", ""),
                "language": payload.get("language", "es"),
                "ner_prompt": payload.get("ner_prompt", ""),
                "validator_prompt": payload.get("validator_prompt", ""),
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:12]
    prompt_id = f"client_{signature}"
    out = {
        "id": prompt_id,
        "vertical_description": payload.get("vertical_description", ""),
        "language": payload.get("language", "es"),
        "ner_prompt": payload.get("ner_prompt", ""),
        "validator_prompt": payload.get("validator_prompt", ""),
        "meta": payload.get("meta", {}),
    }
    path = _presets_dir() / f"{prompt_id}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return prompt_id


def generate_client_prompts(payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    client = VllmChatClient(
        base_url=settings.vllm_base_url,
        model_id=settings.vllm_model_id,
        api_key=settings.vllm_api_key,
        timeout_seconds=settings.vllm_timeout_seconds,
        max_retries=settings.vllm_max_retries,
        disable_thinking=settings.vllm_disable_thinking,
        max_parallel=settings.vllm_max_parallel,
    )

    attempts = ["", "You must comply"]
    parsed: dict[str, Any] | None = None
    for suffix in attempts:
        text = client.complete(
            _build_meta_prompt(payload, strict_suffix=suffix),
            max_tokens=1200,
            temperature=0.2,
            top_p=0.95,
        )
        candidate = extract_first_json_object(text)
        if isinstance(candidate, dict) and _validate_generated_payload(candidate):
            parsed = candidate
            break

    if not parsed:
        raise RuntimeError("No se pudo generar prompts validos para este cliente")

    prompt_id = _save_generated_prompt_bundle({**payload, **parsed})
    payload_hash = hashlib.sha1(
        json.dumps({"ner_prompt": parsed["ner_prompt"], "validator_prompt": parsed["validator_prompt"]}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    logger.info("Prompt bundle generated id=%s hash=%s", prompt_id, payload_hash)

    meta = parsed.get("meta", {}) if isinstance(parsed.get("meta"), dict) else {}
    notes = list(meta.get("notes", [])) if isinstance(meta.get("notes"), list) else []
    notes.append(f"prompt_id={prompt_id}")

    return {
        "ner_prompt": parsed["ner_prompt"],
        "validator_prompt": parsed["validator_prompt"],
        "meta": {
            "assumptions": list(meta.get("assumptions", [])) if isinstance(meta.get("assumptions"), list) else [],
            "recommended_thresholds": meta.get("recommended_thresholds", {"accept": 0.8, "reject": 0.35}),
            "notes": notes,
        },
    }
