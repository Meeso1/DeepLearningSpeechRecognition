from abc import ABC, abstractmethod
import numpy as np
from typing import Any
from Models.TrainingHistory import TrainingHistory
from SpectrogramLoading import PathWithLabel


class ModelBase(ABC):
    @abstractmethod
    def train(
        self,
        train_paths: list[PathWithLabel],
        val_paths: list[PathWithLabel] | None = None,
        epochs: int = 10,
        batch_size: int = 32
    ) -> None:
        pass

    @abstractmethod
    def predict(self, X: list[np.ndarray]) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, test_paths: list[PathWithLabel], batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_history(self) -> TrainingHistory:
        pass

    @abstractmethod
    def get_state_dict(self) -> dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "ModelBase":
        pass
