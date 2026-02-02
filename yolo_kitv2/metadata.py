from __future__ import annotations

from typing import Dict


def load_class_names(metadata_path: str) -> Dict[int, str]:
    """
    Load class names from the project's lightweight `Models/metadata.yaml` format.

    The repo currently stores a simple mapping:

        names:
          0: person
          1: bicycle
          ...

    This function intentionally avoids adding a PyYAML dependency.
    """

    names: Dict[int, str] = {}
    in_names = False

    with open(metadata_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line == "names:":
                in_names = True
                continue
            if not in_names:
                continue

            # Parse "id: label"
            if ":" not in line:
                continue
            left, right = line.split(":", 1)
            left = left.strip()
            right = right.strip().strip("'").strip('"')
            if not left.isdigit():
                continue
            names[int(left)] = right

    return names

