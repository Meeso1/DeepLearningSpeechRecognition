import os
import wave
import sys
import concurrent.futures
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Paths import train_audio_dir, output_dir
from Labels import *
from SerializeToJsonFile import write_to_file

def process_file(args):
    file_path, word = args
    with wave.open(file_path, "rb") as audio:
        framerate = audio.getframerate()
        frames = audio.readframes(framerate)
        
        audio_data = np.frombuffer(frames, dtype=np.int16)
        rms_amplitude = np.sqrt(np.mean(np.square(audio_data, dtype=np.int64)))
        
        return word, rms_amplitude

if __name__ == "__main__":
    # Collect all file paths from all folders
    print("Collecting file paths...")
    file_paths = [(os.path.join(train_audio_dir, folder, file), folder) 
                  for folder in all_folders 
                  for file in os.listdir(os.path.join(train_audio_dir, folder))]

    print("Processing files...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_file, file_paths))

    # Organize results by word
    print("Organizing results by word...")
    rms_by_word = {}
    for label in labels:
        rms_by_word[label] = []
    for word, duration in results:
        if word in unknown_folders:
            rms_by_word["unknown"].append(duration)
        elif word in known_folders:
            rms_by_word[word].append(duration)

    # Write the results to a JSON file
    print("Writing to files...")
    write_to_file(output_dir, "rms_by_word.json", rms_by_word, "RMS Amplitudes by Word")

    print("Processing complete.")
