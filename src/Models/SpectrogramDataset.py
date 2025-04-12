import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from SpectrogramLoading import PathWithLabel


class SpectrogramDataset(Dataset):
    def __init__(self, paths_with_labels: list[PathWithLabel], label_index: dict[str, int]):
        self.paths_with_labels = paths_with_labels
        self.label_index = label_index

    def _to_one_hot(self, label: str) -> np.ndarray:
        one_hot = np.zeros(len(self.label_index))
        one_hot[self.label_index[label]] = 1
        return one_hot

    def __len__(self):
        return len(self.paths_with_labels)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.paths_with_labels[idx]
        img = Image.open(v.path)
        img = np.array(img)
        img = img / 255
        return torch.tensor(img, dtype=torch.float32), torch.tensor(self._to_one_hot(v.label), dtype=torch.float32)