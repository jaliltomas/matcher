import logging
from functools import lru_cache
from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

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
    qwen_ner_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    qwen_validator_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    reranker_model_id: str = "joeddav/xlm-roberta-large-xnli"

    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype: str = "float16"
    batch_size: int = 8
    image_download_workers: int = 24
    image_timeout_seconds: int = 4
    max_new_tokens: int = 128
    offload_between_stages: bool = True

    top_n_default: int = 30
    top_k_default: int = 5


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    if settings.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "DEVICE=%s pero PyTorch no tiene CUDA habilitado. Fallback automatico a CPU.",
            settings.device,
        )
        settings.device = "cpu"

    if settings.device == "cpu" and settings.dtype == "float16":
        logger.warning("DTYPE=float16 en CPU no es recomendable. Se usara float32 en inferencia.")

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
