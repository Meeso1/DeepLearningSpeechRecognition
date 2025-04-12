import wave, os
from pathlib import Path
from Paths import *

input_folder = train_audio_dir / "_background_noise_"
output_folder = train_audio_dir / "silence"

output_folder.mkdir(parents=True, exist_ok=True)

def split_file(input_filename, desired_length = 1):
    input_filepath = os.path.join(input_folder, input_filename)
    with wave.open(input_filepath, "rb") as input_file:
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

            output_filepath = os.path.join(output_folder, f"{input_filename}_{chunk_count}.wav")
            
            with wave.open(output_filepath, "wb") as output_file:
                output_file.setnchannels(num_channels)
                output_file.setsampwidth(sample_width)
                output_file.setframerate(frame_rate)
                output_file.writeframes(chunk_data)

            start_frame = end_frame
            end_frame = min(end_frame + desired_frames_per_second, num_frames)
            chunk_count += 1


def split_files():
    for file in input_folder.iterdir():
        if file.is_file() and file.suffix.lower() == ".wav":
            print(f"Splitting {file.name}...")
            split_file(file.name)

split_files()
print("Processing complete.")