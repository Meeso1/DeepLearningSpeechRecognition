import os, shutil
from concurrent.futures import ThreadPoolExecutor
from Paths import *
from Labels import unknown_folders

output_path = train_audio_dir / "unknown"
output_path.mkdir(parents=True, exist_ok=True)
with ThreadPoolExecutor() as executor:
    for folder in unknown_folders:
        input_path = train_audio_dir / folder
        print(f"Copying from {input_path} to {output_path}")
        for file in os.listdir(input_path):
            executor.submit(shutil.copy, input_path / file, output_path)

print("Processing complete")