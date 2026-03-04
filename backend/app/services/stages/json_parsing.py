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

    def _extract_loose_field(raw_text: str, names: list[str]) -> str | None:
        for name in names:
            patterns = [
                rf'"{name}"\s*:\s*"([^"]+)"',
                rf"'{name}'\s*:\s*'([^']+)'",
                rf'"{name}"\s*:\s*([^,\n\r}}]+)',
                rf"'{name}'\s*:\s*([^,\n\r}}]+)",
                rf"\b{name}\b\s*:\s*([^,\n\r}}]+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, raw_text, re.IGNORECASE)
                if not match:
                    continue
                candidate = match.group(1).strip().strip('"').strip("'")
                cleaned = _clean_scalar(candidate, 80)
                if cleaned is not None:
                    return cleaned
        return None

    if compact["brand"] is None and isinstance(raw_candidate, str):
        compact["brand"] = _extract_loose_field(raw_candidate, ["brand", "marca"])

    if compact["category"] is None and isinstance(raw_candidate, str):
        compact["category"] = _extract_loose_field(raw_candidate, ["category", "categoria"])

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
