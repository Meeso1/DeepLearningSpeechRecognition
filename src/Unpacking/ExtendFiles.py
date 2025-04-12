import wave
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from Constants.Paths import *
from Constants.Labels import all_folders


def extend_file(file_path: str, desired_length: float = 1, noise_level: float = 0.001) -> None:
    with wave.open(file_path, "rb") as file:
        num_channels = file.getnchannels()
        sample_width = file.getsampwidth()
        frame_rate = file.getframerate()
        num_frames = file.getnframes()
        frames = file.readframes(num_frames)

    desired_frames = desired_length * frame_rate
    frames_to_add = desired_frames - num_frames
    dtype = np.int16 if sample_width == 2 else np.int8

    if frames_to_add > 0:        
        noise_frames = np.random.normal(0, noise_level, frames_to_add)
        noise_frames = (noise_frames * (2 ** (8 * sample_width - 1))).astype(dtype).tobytes()
        frames += noise_frames

        with wave.open(file_path, "wb") as file:
            file.setnchannels(num_channels)
            file.setsampwidth(sample_width)
            file.setframerate(frame_rate)
            file.writeframes(frames)


def extend_files() -> None:
    """Extend all audio files to a fixed length by adding low-level noise."""
    tasks = []
    with ThreadPoolExecutor() as executor:
        for folder in all_folders:
            path = train_audio_dir / folder
            if not path.exists():
                print(f"Warning: Folder {path} does not exist. Skipping.")
                continue
            for file in path.iterdir():
                if file.is_file() and file.suffix.lower() == ".wav":
                    tasks.append(executor.submit(extend_file, str(file)))
        
        if test_audio_dir.exists():
            for file in test_audio_dir.iterdir():
                if file.is_file() and file.suffix.lower() == ".wav":
                    tasks.append(executor.submit(extend_file, str(file)))
        else:
            print(f"Warning: Test audio directory {test_audio_dir} does not exist. Skipping.")

        # Wait for all tasks to complete
        for task in tasks:
            task.result()