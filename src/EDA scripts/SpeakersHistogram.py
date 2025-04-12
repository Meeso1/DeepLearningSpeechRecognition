from collections import defaultdict
import os, sys
import multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Paths import train_audio_dir, output_dir
from Labels import *
from SerializeToJsonFile import write_to_file

def count_speakers_in_folder(folder):
    folder_speakers_count = defaultdict(int)
    path = os.path.join(train_audio_dir, folder)
    files = os.listdir(path)
    for file in files:
        hash = file.split("_")[0]
        folder_speakers_count[hash] += 1
    return folder_speakers_count

def merge_dicts(dict_list):
    result = defaultdict(int)
    for d in dict_list:
        for key, value in d.items():
            result[key] += value
    return result

if __name__ == '__main__':
    print("Counting speakers in parallel...")
    num_processes = mp.cpu_count()
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(count_speakers_in_folder, all_folders)
    
    speakers_count = merge_dicts(results)
    
    print("Writing to file...")
    write_to_file(output_dir, "speakers_counts.json", speakers_count, "Counts")

    print("Processing complete.")
