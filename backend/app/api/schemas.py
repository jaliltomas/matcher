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
    extraction_prompt_id: str | None = None
    validation_prompt_id: str | None = None
    extraction_prompt_text: str | None = None
    validation_prompt_text: str | None = None


class MatchCandidate(BaseModel):
    nombre: str
    url: str
    img: str | None
    precio: float
    score_similitud: float
    score_reranker: float
    score_validacion: float
    revisar: bool
    atributos: dict[str, Any]
    sitio: str | None = None
    seller: str | None = None


class AnchorMatches(BaseModel):
    anchor_id: str
    anchor_nombre: str
    atributos_anchor: dict[str, Any]
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
    extraction_prompt_id: str
    validation_prompt_id: str
    results: list[AnchorMatches]
    metrics: list[StageMetric]


class PromptPreset(BaseModel):
    id: str
    name: str
    description: str
    template: str


class PromptPresetResponse(BaseModel):
    extraction: list[PromptPreset]
    validation: list[PromptPreset]
    defaults: dict[str, str]
