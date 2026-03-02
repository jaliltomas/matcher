import json
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
        return text[:max_len]

    raw_attrs = payload.get("atributos", [])
    attributes: list[str] = []
    if isinstance(raw_attrs, list):
        for item in raw_attrs:
            if isinstance(item, str):
                clean = item.strip()
                if clean:
                    attributes.append(clean[:80])
            elif isinstance(item, dict):
                for key, value in item.items():
                    clean_val = _clean_scalar(value, 60)
                    clean_key = _clean_scalar(key, 40)
                    if clean_key and clean_val:
                        attributes.append(f"{clean_key}: {clean_val}"[:90])

    compact = {
        "marca": _clean_scalar(payload.get("marca")),
        "modelo": _clean_scalar(payload.get("modelo")),
        "categoria": _clean_scalar(payload.get("categoria")),
        "atributos": attributes[:10],
        "unidad": _clean_scalar(payload.get("unidad"), 40),
    }

    # Recupera cache vieja o salidas contaminadas tipo "{...}Human: ..."
    # intentando reparsear el primer JSON embebido en raw.
    raw_candidate = payload.get("raw")
    if isinstance(raw_candidate, str):
        has_core = any([compact["marca"], compact["modelo"], compact["categoria"], compact["atributos"], compact["unidad"]])
        if not has_core:
            reparsed = extract_first_json_object(raw_candidate)
            if isinstance(reparsed, dict):
                return normalize_attributes(reparsed, raw_fallback=raw_fallback)

    if raw_fallback and not any([compact["marca"], compact["modelo"], compact["categoria"], compact["atributos"], compact["unidad"]]):
        compact["raw"] = raw_fallback[:180]

    return compact
