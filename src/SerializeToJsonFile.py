import os, json

def write_to_file(output_dir, filename, data, name):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as file:
        json.dump(data, file)
    print(f"{name} saved to: ", filepath)