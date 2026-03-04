from typing import Any

from pydantic import BaseModel, Field


class AnchorItem(BaseModel):
    nombre: str
    img: str | None = None


class ProductItem(BaseModel):
    nombre: str
    url_producto: str
    img: str | None = None
    precioFinal: float
    precioLista: float | None = None
    sitio: str | None = None
    seller: str | None = None
    textoffer: str | None = None
    cantidad: int | None = None


class UploadResponse(BaseModel):
    session_id: str
    anchors_count: int
    products_count: int
    uploaded_files: list[str]


class MatchRequest(BaseModel):
    session_id: str
    top_n: int = Field(default=30, ge=5, le=200)
    top_k: int = Field(default=5, ge=1, le=20)
    batch_size: int | None = Field(default=None, ge=1, le=256)
    ner_batch_size: int | None = Field(default=None, ge=1, le=128)
    validator_batch_size: int | None = Field(default=None, ge=1, le=128)
    use_resume: bool = True
    use_fast_rules: bool = True
    extraction_prompt_id: str | None = None
    validation_prompt_id: str | None = None
    extraction_prompt_text: str | None = None
    validation_prompt_text: str | None = None
    th_accept: float | None = Field(default=None, ge=0.0, le=1.0)
    th_reject: float | None = Field(default=None, ge=0.0, le=1.0)


class NerEvidence(BaseModel):
    brand: str | None = None
    category: str | None = None


class NerAttributes(BaseModel):
    brand: str | None = None
    category: str | None = None
    evidence: NerEvidence = Field(default_factory=NerEvidence)


class MatchCandidate(BaseModel):
    nombre: str
    url: str
    img: str | None
    precio: float
    score_similitud: float
    score_reranker: float
    score_validacion: float
    revisar: bool
    razon_validacion: str | None = None
    cantidad_match: int | None = None
    decision_validacion: str | None = None
    atributos: NerAttributes
    sitio: str | None = None
    seller: str | None = None


class AnchorMatches(BaseModel):
    anchor_id: str
    anchor_nombre: str
    atributos_anchor: NerAttributes
    matches: list[MatchCandidate]


class StageMetric(BaseModel):
    stage: str
    seconds: float
    vram_mb: float
    details: dict[str, Any] = Field(default_factory=dict)


class MatchResponse(BaseModel):
    session_id: str
    top_n: int
    top_k: int
    use_resume: bool
    use_fast_rules: bool | None = None
    extraction_prompt_id: str
    validation_prompt_id: str
    thresholds: dict[str, float] = Field(default_factory=dict)
    results: list[AnchorMatches]
    metrics: list[StageMetric]
    efficiency_stats: dict[str, Any] = Field(default_factory=dict)


class PromptPreset(BaseModel):
    id: str
    name: str
    description: str
    template: str


class PromptPresetResponse(BaseModel):
    extraction: list[PromptPreset]
    validation: list[PromptPreset]
    defaults: dict[str, str]


class PromptGenerateRequest(BaseModel):
    vertical_description: str = Field(min_length=3)
    language: str = Field(default="es", pattern="^(es|en)$")
    brand_notes: str | None = None
    category_taxonomy: list[str] | None = None
    edge_cases: list[str] | None = None
    output_format: str = Field(default="json_only")


class PromptGenerateMeta(BaseModel):
    assumptions: list[str] = Field(default_factory=list)
    recommended_thresholds: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class PromptGenerateResponse(BaseModel):
    ner_prompt: str
    validator_prompt: str
    meta: PromptGenerateMeta
