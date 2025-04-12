import os, wave, sys, concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Paths import train_audio_dir, output_dir
from Labels import all_folders
from SerializeToJsonFile import write_to_file

def process_file(file_path):
    with wave.open(file_path, "rb") as audio:
        framerate = audio.getframerate()
        length = audio.getnframes() / framerate
        sampwidth = audio.getsampwidth() * 8
    return framerate, length, sampwidth

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

    # Separate the framerates, lengths, and sample widths
    print("Separating results...")
    framerates, lengths, sampwidths = map(list, zip(*results))

    # Write the results to files
    print("Writing to files...")
    write_to_file(output_dir, "framerates.json", framerates, "Framerates")
    write_to_file(output_dir, "lengths.json", lengths, "Lengths")
    write_to_file(output_dir, "sampwidths.json", sampwidths, "SampleWidths")

    print("Processing complete.")