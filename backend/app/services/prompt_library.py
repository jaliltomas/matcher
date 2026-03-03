import hashlib
from typing import Literal


PromptKind = Literal["extraction", "validation"]


EXTRACTION_PRESETS = [
    {
        "id": "g2_bebidas_extract_v1",
        "name": "G2 Bebidas",
        "description": "Extractor especializado en bebidas: volumen unitario, envase, variante y pack.",
        "template": (
            "Eres un extractor experto de atributos para productos de BEBIDAS en eCommerce. "
            "Responde SOLO JSON valido (sin markdown ni texto extra) con estructura exacta: "
            '{"marca":null,"modelo":null,"categoria":null,"atributos":[],"unidad":null}. '
            "Extrae y normaliza, solo desde el nombre del producto: "
            "marca, linea/variedad, tipo de bebida, sabor/variante, graduacion alcoholica (si aparece), "
            "volumen por unidad (ml/L/cc), tipo de envase (lata/botella/vidrio/plastico), cantidad de unidades, "
            "y si incluye estuche/regalo/accesorios. "
            "Reglas: (1) No inventes. (2) Si falta evidencia, usar null. "
            "(3) En 'atributos' incluir pares cortos tipo 'tipo: cerveza', 'variante: ipa', 'pack: x6', 'envase: lata'. "
            "(4) En 'unidad' poner SIEMPRE el volumen por unidad normalizado (ej: '473 ml', '750 ml', '1 l') si existe. "
            "(5) No mezclar pack con unidad: pack va en atributos, unidad es solo volumen unitario. "
            "(6) Si detectas estuche/regalo/set/kit/+ vaso/+ copa, incluirlo en atributos. "
            "Texto producto: {TEXT}"
        ),
    },
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
        "id": "g2_bebidas_v1",
        "name": "G2 Bebidas",
        "description": "Validator estricto para bebidas; tolera cambio de pack pero no de volumen por unidad.",
        "template": (
            "Rol: analista experto en PRODUCT MATCHING DE BEBIDAS para eCommerce. "
            "Objetivo: decidir si cada candidato es EXACTAMENTE la misma bebida que el ancla. "
            "Regla de oro: en caso de duda, RECHAZAR. Aceptar solo con confianza >=90%. "
            "Atributos criticos que deben coincidir: marca, linea/variedad, tipo de bebida, sabor/variante, "
            "graduacion alcoholica (si aparece), volumen por unidad, y presencia de estuche/regalo/accesorios. "
            "Si cambia cualquier atributo critico => rechazar. "
            "Cantidad de unidades (x1,x2,x6,x12, pack) es flexible y NO invalida match. "
            "Volumen por unidad NO es flexible. "
            "Si un atributo viene null o ausente en ATRIBUTOS, NO rechaces automaticamente: usa evidencia del NOMBRE. "
            "La ausencia de modelo explicito en candidato NO implica rechazo si nombre+marca+tipo+unidad coinciden fuerte. "
            "Sublineas/labels distintos (red/black/blue, reserva, anejo, edition, select, single malt, etc.) => rechazar. "
            "Productos con gift box, estuche, set, kit, con vaso/copa/hielera/accesorio => producto distinto => rechazar. "
            "No uses precio como criterio principal. "
            "Debes responder SOLO JSON valido, sin markdown ni texto extra, con formato exacto: "
            '{"matches_validos":[{"id":1,"razon":"texto_corto","cantidad":1}],"rechazados":[{"id":2,"razon":"texto_corto"}]}. '
            "Reglas de salida obligatorias: "
            "(1) incluir TODOS los ids de candidatos exactamente una vez; "
            "(2) no inventar ids; "
            "(3) no repetir ids; "
            "(4) en matches_validos siempre incluir cantidad>=1; "
            "(5) razon breve (<80 chars en validos, <100 en rechazados). "
            "Cantidad: detectar xN/pack N/six pack; si no aparece, usar 1. "
            "Si no hay matches, devolver matches_validos vacio y todos en rechazados. "
            "ANCLA={ANCHOR_NAME}\n"
            "ATRIBUTOS_ANCLA={ANCHOR_ATTRS}\n"
            "CANDIDATOS={CANDIDATES_JSON}"
        ),
    },
    {
        "id": "strict_identity_v1",
        "name": "Strict Identity",
        "description": "Exige coincidencia fuerte de marca/modelo/formato.",
        "template": (
            "Evalua candidatos vs ancla con criterio estricto de identidad de SKU. "
            "Si hay diferencia relevante en marca/modelo/formato/presentacion, rechazar. "
            "Responde SOLO JSON valido (sin texto extra) con formato: "
            '{"matches_validos":[{"id":1,"razon":"","cantidad":1}],"rechazados":[{"id":2,"razon":""}]}. '
            "Incluye todos los ids. "
            "ANCLA={ANCHOR_NAME}\n"
            "ATRIBUTOS_ANCLA={ANCHOR_ATTRS}\n"
            "CANDIDATOS={CANDIDATES_JSON}"
        ),
    },
    {
        "id": "category_tolerant_v1",
        "name": "Category Tolerant",
        "description": "Tolera variaciones menores de naming dentro de misma categoria.",
        "template": (
            "Valida candidatos retail con tolerancia moderada de naming. "
            "Prioriza marca + categoria + unidad/formato y rechaza cambios de variante/modelo. "
            "Responde SOLO JSON valido con formato: "
            '{"matches_validos":[{"id":1,"razon":"","cantidad":1}],"rechazados":[{"id":2,"razon":""}]}. '
            "Incluye todos los ids. "
            "ANCLA={ANCHOR_NAME}\n"
            "ATRIBUTOS_ANCLA={ANCHOR_ATTRS}\n"
            "CANDIDATOS={CANDIDATES_JSON}"
        ),
    },
    {
        "id": "price_guard_v1",
        "name": "Price Guard",
        "description": "Incluye chequeo de coherencia por precio y pack.",
        "template": (
            "Determina si cada candidato es el mismo producto o equivalente exacto de pack. "
            "Usa precio como senal secundaria, no principal. "
            "Responde SOLO JSON valido con formato: "
            '{"matches_validos":[{"id":1,"razon":"","cantidad":1}],"rechazados":[{"id":2,"razon":""}]}. '
            "Incluye todos los ids. "
            "ANCLA={ANCHOR_NAME}\n"
            "ATRIBUTOS_ANCLA={ANCHOR_ATTRS}\n"
            "CANDIDATOS={CANDIDATES_JSON}"
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
