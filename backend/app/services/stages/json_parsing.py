import json
import re
from typing import Any


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if "```" in candidate:
        candidate = candidate.replace("```json", "").replace("```", "").strip()

    start = candidate.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False

        for idx in range(start, len(candidate)):
            ch = candidate[idx]

            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    raw = candidate[start : idx + 1]
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        break

        start = candidate.find("{", start + 1)

    return None


def normalize_attributes(payload: dict[str, Any], raw_fallback: str | None = None) -> dict[str, Any]:
    def _clean_scalar(value: Any, max_len: int = 80):
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"null", "none", "n/a", "na", "-", "s/d", "sin dato"}:
            return None
        return text[:max_len]

    raw_brand = payload.get("brand") if "brand" in payload else payload.get("marca")
    raw_category = payload.get("category") if "category" in payload else payload.get("categoria")
    raw_evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}

    compact = {
        "brand": _clean_scalar(raw_brand),
        "category": _clean_scalar(raw_category),
        "evidence": {
            "brand": _clean_scalar(raw_evidence.get("brand"), 160),
            "category": _clean_scalar(raw_evidence.get("category"), 160),
        },
    }

    raw_candidate = payload.get("raw")
    if compact["brand"] is None and isinstance(raw_candidate, str):
        brand_match = re.search(r'"(?:brand|marca)"\s*:\s*"([^"]+)"', raw_candidate, re.IGNORECASE)
        if brand_match:
            compact["brand"] = _clean_scalar(brand_match.group(1), 80)

    if compact["category"] is None and isinstance(raw_candidate, str):
        category_match = re.search(r'"(?:category|categoria)"\s*:\s*"([^"]+)"', raw_candidate, re.IGNORECASE)
        if category_match:
            compact["category"] = _clean_scalar(category_match.group(1), 80)

    # Recupera cache vieja o salidas contaminadas tipo "{...}Human: ..."
    # intentando reparsear el primer JSON embebido en raw.
    if isinstance(raw_candidate, str):
        has_core = any([compact["brand"], compact["category"]])
        if not has_core:
            reparsed = extract_first_json_object(raw_candidate)
            if isinstance(reparsed, dict):
                return normalize_attributes(reparsed, raw_fallback=raw_fallback)

    if raw_fallback and not any([compact["brand"], compact["category"]]):
        compact["raw"] = raw_fallback[:180]

    return compact
