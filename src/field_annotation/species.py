from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def load_species_info(path: Path) -> dict:
    return json.loads(Path(path).read_text())["species"]


def get_class_id(species_info: dict, species_name: Optional[str]) -> Optional[int]:
    """Case-insensitive match against common_name, then alias list. Adapted
    from the old build_metadata.py::MetadataExtractor._find_class_id,
    simplified to take a species name directly (from file_status.species)
    instead of looking it up via a CSV-row join.
    """
    if not species_name:
        return None
    needle = species_name.strip().lower()
    for entry in species_info.values():
        if entry.get("common_name", "").lower() == needle:
            return entry["class_id"]
        for alias in entry.get("alias") or []:
            if alias.lower() == needle:
                return entry["class_id"]
    return None
