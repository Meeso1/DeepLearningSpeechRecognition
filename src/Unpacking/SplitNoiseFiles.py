import wave
import os
from Constants.Paths import *


def split_file(input_path: Path, output_folder: Path, desired_length: int = 1) -> None:
    input_filename = input_path.name
    
    with wave.open(str(input_path), "rb") as input_file:
        num_channels = input_file.getnchannels()
        sample_width = input_file.getsampwidth()
        frame_rate = input_file.getframerate()
        num_frames = input_file.getnframes()

        desired_frames_per_second = desired_length * frame_rate * num_channels
        start_frame = 0
        end_frame = desired_frames_per_second

        chunk_count = 0
        while start_frame < num_frames:
            input_file.setpos(start_frame)
            chunk_data = input_file.readframes(end_frame - start_frame)

            input_filename_without_extension = os.path.splitext(input_filename)[0]
            output_filepath = os.path.join(output_folder, f"{input_filename_without_extension}_{chunk_count}.wav")
            
            with wave.open(output_filepath, "wb") as output_file:
                output_file.setnchannels(num_channels)
                output_file.setsampwidth(sample_width)
                output_file.setframerate(frame_rate)
                output_file.writeframes(chunk_data)

            start_frame = end_frame
            end_frame = min(end_frame + desired_frames_per_second, num_frames)
            chunk_count += 1


def split_files(length: float = 1, skip_if_output_exists: bool = True) -> None:
    """Split noise files into 1-second chunks and save them in the silence folder."""
    
    input_folder = train_audio_dir / "_background_noise_"
    output_folder = train_audio_dir / "silence"
    
    if skip_if_output_exists and output_folder.exists():
        print(f"Silence folder {output_folder} already exists. Skipping.")
        return
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for file in input_folder.iterdir():
        if file.is_file() and file.suffix.lower() == ".wav":
            split_file(file, output_folder, desired_length=length)
