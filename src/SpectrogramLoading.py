import numpy as np
from PIL import Image
from Paths import train_spectrograms_dir, test_spectrograms_dir, output_dir_absolute_path
import os
from dataclasses import dataclass
from Labels import known_folders, labels


@dataclass
class PathWithLabel:
    path: str
    label: str


@dataclass
class SpectrogramWithLabel:
    """Spectrogram has shape (n_mels, time_steps) = (128, ?)"""
    spectrogram: np.ndarray
    label: str
    

def get_train_validation_relative_paths() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    with open(output_dir_absolute_path / "training_set.txt", "r") as f:
        train_paths = [line.strip().replace("\\", os.path.sep) for line in f.readlines()]
    with open(output_dir_absolute_path / "validation_set.txt", "r") as f:
        validation_paths = [line.strip().replace("\\", os.path.sep) for line in f.readlines()]
        
    return train_paths, validation_paths

def get_test_paths_with_labels() -> list[PathWithLabel]:
    with open(output_dir_absolute_path / "testing_set.csv", "r") as f:
        next(f)
        path, label = zip(*(line.strip().split(',') for line in f))
    
    path = [p.replace(".wav", ".png") for p in path]
    return [PathWithLabel(test_spectrograms_dir / p, l) for p, l in zip(path, label)]
    

def get_paths_by_label() -> dict[str, list[str]]:
    """Get the paths of the spectrograms by label."""
    paths_by_label = {}
    
    # Iterate through each label folder in the train_spectograms_dir
    for label_folder in os.listdir(train_spectrograms_dir):
        folder_path = train_spectrograms_dir / label_folder
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue
            
        # Get all PNG files in this label folder
        spectrogram_files = [
            str(folder_path / filename) 
            for filename in os.listdir(folder_path) 
            if filename.endswith('.png')
        ]
        
        # Add to dictionary
        paths_by_label[label_folder] = spectrogram_files
    
    return paths_by_label


def get_divided_paths_with_labels() -> tuple[list[PathWithLabel], list[PathWithLabel]]:
    train_paths, validation_paths = get_train_validation_relative_paths()
    
    return [PathWithLabel(train_spectrograms_dir / path, path.split("/")[0] if path.split("/")[0] in known_folders else labels[-1]) for path in train_paths], \
        [PathWithLabel(train_spectrograms_dir / path, path.split("/")[0] if path.split("/")[0] in known_folders else labels[-1]) for path in validation_paths]


def to_paths_with_labels(paths_by_label: dict[str, list[str]]) -> list[PathWithLabel]:
    return [PathWithLabel(path, label) for label, paths in paths_by_label.items() for path in paths]


def load_spectrogram_from_png_file(file_path: str) -> np.ndarray:
    """Load a spectrogram from a PNG file."""
    with Image.open(file_path) as image:
        pixel_array = np.array(image)
    pixel_array = pixel_array / 255.0
    return pixel_array


def load_spectrogram_from_path(path_with_label: PathWithLabel) -> SpectrogramWithLabel:
    """Load a spectrogram from a path with label."""
    spectrogram = load_spectrogram_from_png_file(path_with_label.path)
    return SpectrogramWithLabel(spectrogram, path_with_label.label)

def spectrograms_to_x_y(
    spectrograms: list[SpectrogramWithLabel],
    label_indexes: dict[str, int] | None = None
) -> tuple[list[np.ndarray], np.ndarray, dict[str, int]]:
    if label_indexes is None:
        label_indexes = {v: i for i, v in enumerate(sorted(set([v.label for v in spectrograms])))}

    X = [v.spectrogram for v in spectrograms]
    y = np.array([label_indexes[v.label] for v in spectrograms])
    return X, y, label_indexes
