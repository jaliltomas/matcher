import hashlib
from typing import Literal


PromptKind = Literal["extraction", "validation"]


EXTRACTION_PRESETS = [
    {
        "id": "retail_general_v1",
        "name": "Retail General",
        "description": "Balanceado para catalogo mixto de consumo masivo.",
        "template": (
            "Tarea: extraer atributos para matching de productos retail. "
            "Responde SOLO JSON valido (sin markdown ni texto extra) con estructura exacta: "
            '{"marca":null,"modelo":null,"categoria":null,"atributos":[],"unidad":null}. '\
            "Reglas: (1) No inventes datos. (2) Usa null si falta evidencia. "
            "(3) atributos debe ser lista corta y util (tamano, sabor, color, pack, capacidad). "
            "(4) unidad debe ser por ejemplo '1 kg', '750 ml', 'pack x6' si aparece. "
            "(5) Nunca incluyas ejemplos, dialogos, ni prefijos como Human/Assistant. "
            "Texto producto: {TEXT}"
        ),
    },
    {
        "id": "electronics_v1",
        "name": "Electronics",
        "description": "Prioriza modelo tecnico, capacidad y compatibilidad.",
        "template": (
            "Extrae atributos para productos de electronica. "
            "Responde SOLO JSON valido (sin texto extra) con: "
            '{"marca":null,"modelo":null,"categoria":null,"atributos":[],"unidad":null}. '\
            "Incluye en atributos: almacenamiento, memoria, color, version, conectividad, compatibilidad. "
            "Si no aparece, no adivines. Prohibido responder con dialogo o ejemplos. Texto: {TEXT}"
        ),
    },
    {
        "id": "grocery_v1",
        "name": "Grocery",
        "description": "Optimizado para alimentos, bebidas y limpieza.",
        "template": (
            "Extrae entidades para productos de supermercado. "
            "Devuelve SOLO JSON valido (sin texto adicional): "
            '{"marca":null,"modelo":null,"categoria":null,"atributos":[],"unidad":null}. '\
            "En atributos prioriza: variedad/sabor, formato, tamano, pack, tipo de producto. "
            "No agregues ningun texto fuera del JSON. "
            "Texto: {TEXT}"
        ),
    },
]


VALIDATION_PRESETS = [
    {
        "id": "strict_identity_v1",
        "name": "Strict Identity",
        "description": "Exige coincidencia fuerte de marca/modelo/formato.",
        "template": (
            "Evalua si ANCLA y CANDIDATO son el mismo SKU comercial. "
            "Devuelve SOLO JSON valido (sin texto extra) con formato exacto: "
            '{"validation_score":0.0,"review_flag":false,"reason":""}. '\
            "Criterio: marca/modelo/formato/presentacion deben coincidir. Diferencia fuerte => score bajo. "
            "validation_score entre 0 y 1. review_flag=true si score<0.70 o hay ambiguedad. "
            "No escribas ejemplos, ni Human/Assistant. "
            "ANCLA={ANCHOR_NAME}\n"
            "ATRIBUTOS_ANCLA={ANCHOR_ATTRS}\n"
            "CANDIDATO={CANDIDATE_NAME}\n"
            "ATRIBUTOS_CANDIDATO={CANDIDATE_ATTRS}\n"
            "URL={URL}\n"
            "PRECIO={PRICE}"
        ),
    },
    {
        "id": "category_tolerant_v1",
        "name": "Category Tolerant",
        "description": "Tolera variaciones menores de naming dentro de misma categoria.",
        "template": (
            "Valida matching de productos retail con tolerancia moderada a nombres comerciales. "
            "Responde SOLO JSON valido (sin texto fuera del JSON): "
            '{"validation_score":0.0,"review_flag":false,"reason":""}. '\
            "Prioriza coincidencia de marca + categoria + unidad/formato. "
            "Penaliza cuando cambia sabor, tamano o modelo. "
            "ANCLA={ANCHOR_NAME}\n"
            "ATRIBUTOS_ANCLA={ANCHOR_ATTRS}\n"
            "CANDIDATO={CANDIDATE_NAME}\n"
            "ATRIBUTOS_CANDIDATO={CANDIDATE_ATTRS}\n"
            "URL={URL}\n"
            "PRECIO={PRICE}"
        ),
    },
    {
        "id": "price_guard_v1",
        "name": "Price Guard",
        "description": "Incluye chequeo de coherencia por precio y pack.",
        "template": (
            "Determina si ANCLA y CANDIDATO representan el mismo producto o equivalente exacto de pack. "
            "Responde SOLO JSON valido (sin texto adicional): "
            '{"validation_score":0.0,"review_flag":false,"reason":""}. '\
            "Si unidad o pack difiere, reduce score. Usa precio como senal secundaria, no principal. "
            "ANCLA={ANCHOR_NAME}\n"
            "ATRIBUTOS_ANCLA={ANCHOR_ATTRS}\n"
            "CANDIDATO={CANDIDATE_NAME}\n"
            "ATRIBUTOS_CANDIDATO={CANDIDATE_ATTRS}\n"
            "URL={URL}\n"
            "PRECIO={PRICE}"
        ),
    },
]


def list_prompt_presets() -> dict:
    return {
        "extraction": EXTRACTION_PRESETS,
        "validation": VALIDATION_PRESETS,
        "defaults": {
            "extraction": EXTRACTION_PRESETS[0]["id"],
            "validation": VALIDATION_PRESETS[0]["id"],
        },
    }


def _pick_preset(kind: PromptKind, preset_id: str | None) -> dict:
    presets = EXTRACTION_PRESETS if kind == "extraction" else VALIDATION_PRESETS
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
