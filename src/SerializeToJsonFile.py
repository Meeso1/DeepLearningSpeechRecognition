import os
import json
from typing import Any


def write_to_file(output_dir: str, filename: str, data: Any, name: str) -> None:
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as file:
        json.dump(data, file)

    print(f"{name} saved to: {filepath}")
