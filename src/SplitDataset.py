import hashlib
import os
from Paths import train_spectrograms_dir, output_dir
from Labels import *
from concurrent.futures import ThreadPoolExecutor

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def determine_set(file_path, training_percentage):
    base_name = os.path.basename(file_path)
    hash = hashlib.sha1(base_name.split("_")[0].encode('utf-8')).hexdigest()
    
    percentage = int(hash, 16) % (MAX_NUM_WAVS_PER_CLASS + 1) * (100.0 / MAX_NUM_WAVS_PER_CLASS)
    return "training" if percentage < training_percentage else "validation"

def split_dataset(input_dir, output_dir, training_percentage=80):
    def process_file(file_path):
        set_name = determine_set(file_path, training_percentage)
        
        parts = file_path.split(os.path.sep)
        tail_path = os.path.sep.join(parts[-2:]) if len(parts) > 1 else file_path
        
        return f"{tail_path}", set_name
    
    training_set = []
    validation_set = []

    with ThreadPoolExecutor() as executor:
        for folder in all_folders:
            path = os.path.join(input_dir, folder)
            files = os.listdir(path)
            full_paths = [os.path.join(path, file) for file in files]
            results = executor.map(process_file, full_paths)

            for result in results:
                if result[1] == "training":
                    training_set.append(result[0])
                else:
                    validation_set.append(result[0])
    
    def write_set(file_name, data):
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "w") as file:
            for item in data:
                file.write(f"{item}\n")
        print(f"Set saved to: {file_path}")

    with ThreadPoolExecutor(max_workers=2) as exec_writer:
        future_training = exec_writer.submit(write_set, "training_set.txt", training_set)
        future_validation = exec_writer.submit(write_set, "validation_set.txt", validation_set)
        future_training.result()
        future_validation.result()

split_dataset(train_spectrograms_dir, output_dir)
print("Processing complete.")
