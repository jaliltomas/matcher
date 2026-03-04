import json
import logging
from functools import lru_cache

from fastapi import APIRouter, File, HTTPException, UploadFile
from pymilvus.exceptions import MilvusException
from starlette.concurrency import run_in_threadpool

from app.api.schemas import (
    AnchorItem,
    PromptGenerateRequest,
    PromptGenerateResponse,
    MatchRequest,
    MatchResponse,
    ProductItem,
    PromptPresetResponse,
    UploadResponse,
)
from app.services.data_store import session_store
from app.services.milvus_client import MilvusVectorStore
from app.services.pipeline import MatchingPipeline
from app.services.prompt_library import generate_client_prompts, list_prompt_presets, resolve_prompt

logger = logging.getLogger(__name__)

router = APIRouter()


@lru_cache(maxsize=1)
def get_pipeline() -> MatchingPipeline:
    vector_store = MilvusVectorStore()
    return MatchingPipeline(vector_store=vector_store)


async def _read_json_array(upload_file: UploadFile) -> list[dict]:
    content = await upload_file.read()
    try:
        parsed = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"JSON invalido en {upload_file.filename}: {exc}") from exc

    if not isinstance(parsed, list):
        raise HTTPException(status_code=400, detail=f"El archivo {upload_file.filename} debe contener un array JSON")
    return parsed


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/prompt-presets", response_model=PromptPresetResponse)
def prompt_presets() -> PromptPresetResponse:
    return PromptPresetResponse.model_validate(list_prompt_presets())


@router.post("/prompts/generate", response_model=PromptGenerateResponse)
def generate_prompts(request: PromptGenerateRequest) -> PromptGenerateResponse:
    try:
        generated = generate_client_prompts(request.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return PromptGenerateResponse.model_validate(generated)


@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    anchor_file: UploadFile = File(...),
    price_files: list[UploadFile] = File(...),
) -> UploadResponse:
    anchor_payload = await _read_json_array(anchor_file)
    price_payloads: list[dict] = []
    uploaded_files = [anchor_file.filename or "anchors.json"]

    for file in price_files:
        uploaded_files.append(file.filename or "prices.json")
        file_content = await _read_json_array(file)
        price_payloads.extend(file_content)

    anchors = [AnchorItem.model_validate(item).model_dump() for item in anchor_payload]
    products = [ProductItem.model_validate(item).model_dump() for item in price_payloads]

    if not anchors:
        raise HTTPException(status_code=400, detail="No se encontraron anclas en el archivo")
    if not products:
        raise HTTPException(status_code=400, detail="No se encontraron productos en los JSON de precio")

    session_id = session_store.create_session(anchors, products, uploaded_files)
    logger.info("Upload OK | session=%s anchors=%d products=%d", session_id, len(anchors), len(products))

    return UploadResponse(
        session_id=session_id,
        anchors_count=len(anchors),
        products_count=len(products),
        uploaded_files=uploaded_files,
    )


@router.post("/match", response_model=MatchResponse)
async def run_match(request: MatchRequest) -> MatchResponse:
    session_data = session_store.get_session(request.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session no encontrada. Sube los archivos primero.")

    extraction_prompt, extraction_prompt_id = resolve_prompt(
        kind="extraction",
        preset_id=request.extraction_prompt_id,
        custom_prompt=request.extraction_prompt_text,
    )
    validation_prompt, validation_prompt_id = resolve_prompt(
        kind="validation",
        preset_id=request.validation_prompt_id,
        custom_prompt=request.validation_prompt_text,
    )

    pipeline = get_pipeline()
    try:
        response = await run_in_threadpool(
            pipeline.process,
            session_data,
            request.top_n,
            request.top_k,
            request.batch_size,
            request.ner_batch_size,
            request.validator_batch_size,
            request.use_resume,
            request.use_fast_rules,
            extraction_prompt,
            validation_prompt,
            extraction_prompt_id,
            validation_prompt_id,
            request.th_accept,
            request.th_reject,
        )
    except MilvusException as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Milvus no disponible. Si estas resumiendo, activa use_resume y usa una session_id con cache de vector_search; "
                "si no, levanta Milvus en 127.0.0.1:19530."
            ),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return MatchResponse.model_validate(response)
