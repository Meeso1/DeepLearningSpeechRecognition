import wave, os, sys, concurrent.futures
import numpy as np
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Paths import train_audio_dir, output_dir
from Labels import all_folders
from SerializeToJsonFile import write_to_file

def process_file(file_path):
    with wave.open(file_path, "rb") as audio:
        framerate = audio.getframerate()
        frames = audio.readframes(framerate)
        
        audio_data = np.frombuffer(frames, dtype=np.int16)
        rms_amplitude = np.sqrt(np.mean(np.square(audio_data, dtype=np.int64)))
        
        return rms_amplitude
    
if __name__ == "__main__":
    # Collect all file paths from all folders
    print("Collecting file paths...")
    file_paths = []
    for folder in all_folders:
        path = os.path.join(train_audio_dir, folder)
        for file in os.listdir(path):
            file_paths.append(os.path.join(path, file))

    # Process files concurrently
    print("Processing files...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_file, file_paths))

    # Write the results to files
    print("Writing to files...")
    write_to_file(output_dir, "rms_amplitudes.json", results, "RMSAmplitudes")

    print("Processing complete.")
    