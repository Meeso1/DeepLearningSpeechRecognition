from Unpacking.SplitNoiseFiles import split_files
from Unpacking.ExtendFiles import extend_files
from Unpacking.SplitDataset import split_dataset


def prepare_audio_files(training_percentage: float = 80, extend_to_length: float = 1, noise_level: float = 0.001) -> None:
    split_files(length=extend_to_length, skip_if_output_exists=True)
    extend_files(desired_length=extend_to_length, noise_level=noise_level, skip_if_output_exists=True)
    split_dataset(training_percentage=training_percentage)
