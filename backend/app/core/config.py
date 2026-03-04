import logging
from functools import lru_cache
from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "TurboMatcher API"
    api_prefix: str = ""
    data_dir: Path = Path("./data")

    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_db_name: str = "default"
    milvus_collection_prefix: str = "turbomatcher"

    blip2_model_id: str = "Salesforce/blip2-opt-2.7b"
    qwen_ner_model_id: str = "qwen/qwen3.5-9b"
    qwen_validator_model_id: str = "qwen/qwen3.5-9b"
    use_vllm: bool = False
    vllm_base_url: str = "http://127.0.0.1:8008/v1"
    vllm_api_key: str = "EMPTY"
    vllm_model_id: str = "meta-llama-3.1-8b-instruct"
    vllm_ner_model_id: str = ""
    vllm_validator_model_id: str = ""
    vllm_timeout_seconds: int = 120
    vllm_max_retries: int = 2
    vllm_disable_thinking: bool = True
    vllm_max_parallel: int = 8
    vllm_context_window: int = 4096
    vllm_context_reserve: int = 128
    reranker_model_id: str = "joeddav/xlm-roberta-large-xnli"

    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype: str = "float16"
    batch_size: int = 8
    image_download_workers: int = 24
    image_timeout_seconds: int = 4
    max_new_tokens: int = 4096
    ner_max_new_tokens: int = 1000
    validator_max_new_tokens: int = 256
    offload_between_stages: bool = True

    top_n_default: int = 30
    top_k_default: int = 5
    th_accept: float = 0.80
    th_reject: float = 0.35


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    if not settings.vllm_ner_model_id.strip():
        settings.vllm_ner_model_id = settings.vllm_model_id
    if not settings.vllm_validator_model_id.strip():
        settings.vllm_validator_model_id = settings.vllm_model_id
    if settings.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "DEVICE=%s pero PyTorch no tiene CUDA habilitado. Fallback automatico a CPU.",
            settings.device,
        )
        settings.device = "cpu"

    if settings.device == "cpu" and settings.dtype == "float16":
        logger.warning(
            "DTYPE=float16 en CPU no es recomendable. Se usara float32 en inferencia."
        )

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
