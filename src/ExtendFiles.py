import wave
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from Paths import *
from Labels import all_folders

def extend_file(file_path, desired_length=1, noise_level=0.001):
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

def extend_files():
    tasks = []
    with ThreadPoolExecutor() as executor:
        print("Processing training set")
        for folder in all_folders:
            print(f"Processing {folder}...")
            path = train_audio_dir / folder
            for file in path.iterdir():
                if file.is_file() and file.suffix.lower() == ".wav":
                    tasks.append(executor.submit(extend_file, str(file)))
        print("Processing testing set")
        for file in test_audio_dir.iterdir():
            if file.is_file() and file.suffix.lower() == ".wav":
                tasks.append(executor.submit(extend_file, str(file)))
        # Wait for all tasks to complete
        for task in tasks:
            task.result()

extend_files()
print("Processing complete.")