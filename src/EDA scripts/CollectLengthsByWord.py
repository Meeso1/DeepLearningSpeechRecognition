import os
import wave
import sys
import concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Paths import train_audio_dir, output_dir
from Labels import *
from SerializeToJsonFile import write_to_file

def process_file(args):
    file_path, word = args
    with wave.open(file_path, "rb") as audio:
        framerate = audio.getframerate()
        length = audio.getnframes() / framerate 
    return word, length  

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
    lengths_by_word = {}
    for label in labels:
        lengths_by_word[label] = []
    for word, duration in results:
        if word in unknown_folders:
            lengths_by_word["unknown"].append(duration)
        elif word in known_folders:
            lengths_by_word[word].append(duration)

    # Write the results to a JSON file
    print("Writing to files...")
    write_to_file(output_dir, "lengths_by_word.json", lengths_by_word, "Lengths by Word")

    print("Processing complete.")
