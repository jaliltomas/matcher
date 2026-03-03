from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class EmbeddingStage(ABC):
    @abstractmethod
    def embed_records(self, records: list[dict[str, Any]], batch_size: int) -> np.ndarray:
        raise NotImplementedError


class EnricherStage(ABC):
    @abstractmethod
    def extract_attributes(
        self,
        items: list[dict[str, Any]],
        batch_size: int,
        prompt_template: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        raise NotImplementedError


class RerankerStage(ABC):
    @abstractmethod
    def score_pairs(self, pairs: list[dict[str, Any]], batch_size: int) -> list[float]:
        raise NotImplementedError


class ValidatorStage(ABC):
    @abstractmethod
    def validate_groups(
        self,
        groups: list[dict[str, Any]],
        batch_size: int,
        prompt_template: str | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError
