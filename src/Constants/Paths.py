import os
from pathlib import Path


root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
dataset_dir = root_dir / "Dataset"
train_dir = dataset_dir / "train"
test_dir = dataset_dir / "test"
train_audio_dir = train_dir / "audio"
test_audio_dir = test_dir / "audio"
output_dir = root_dir / "Outputs"
output_dir_absolute_path = root_dir / "Outputs"
train_spectrograms_dir = train_dir / "spectrograms"
test_spectrograms_dir = test_dir / "spectrograms"
models_dir = root_dir / "Models"
