import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Paths import train_audio_dir, output_dir
from Labels import *
from SerializeToJsonFile import write_to_file

counts = {}
for label in labels:
    counts[label] = 0

print("Counting files...")
for folder in known_folders:
    path = os.path.join(train_audio_dir, folder)
    counts[folder] = len(os.listdir(path))

for folder in unknown_folders:
    path = os.path.join(train_audio_dir, folder)
    counts["unknown"] += len(os.listdir(path))


print("Writing to file...")
write_to_file(output_dir, "labels_counts.json", counts, "Counts")

print("Processing complete.")
