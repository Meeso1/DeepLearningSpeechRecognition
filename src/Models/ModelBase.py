from abc import ABC, abstractmethod
import os
import pickle
import shutil
import tempfile
import numpy as np
from typing import Any
import torch
import wandb

from Models.TrainingHistory import TrainingHistory


class ModelBase(ABC):
    @abstractmethod
    def train(
        self,
        train_data: tuple[list[np.ndarray], np.ndarray],
        val_data: tuple[list[np.ndarray], np.ndarray] | None = None,
        epochs: int = 10,
        batch_size: int = 32
    ) -> None:
        pass

    @abstractmethod
    def predict(self, X: list[np.ndarray]) -> np.ndarray:
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
    
    @abstractmethod
    def get_config_for_wandb(self) -> dict[str, Any]:
        pass
    
    # Helper functions
    def set_random_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            
        np.random.seed(seed)
    
    def _from_one_hot(self, y: np.ndarray) -> np.ndarray:
        return np.argmax(y, axis=1)

    def _to_one_hot(self, y: np.ndarray) -> np.ndarray:
        return np.eye(len(np.unique(y)))[y]
    
    def save_model_to_wandb(self, name: str) -> None:
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "model.pkl")
        
        with open(temp_file_path, "wb") as f:
            pickle.dump(self.get_state_dict(), f)
        
        artifact = wandb.Artifact(name=name, type="model", description="Model state dict")
        artifact.add_file(temp_file_path)
        
        logged_artifact = wandb.log_artifact(artifact)
        logged_artifact.wait()
        
        shutil.rmtree(temp_dir)
