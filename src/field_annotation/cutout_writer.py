from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# Filenames/format match the old pipeline's UNetInference.save_image exactly,
# since batches already on LTS use this convention
# (e.g. field-batches/AL_2023-08-17/cutouts/ALA00081_0.jpg / _0_mask.png / _0.png / _0.json).
_JPEG_QUALITY = 100


def build_class_mask(bin_mask: np.ndarray, class_id: int | None) -> np.ndarray:
    """Burns class_id (or 1 as a placeholder when species couldn't be
    resolved) into mask pixel values, same convention as the old pipeline."""
    fg_value = class_id if class_id is not None else 1
    return np.where(bin_mask == 1, fg_value, 0).astype(np.uint8)


def write_cutout_files(
    cutouts_dir: Path,
    base_name: str,
    crop_bgr: np.ndarray,
    mask_class: np.ndarray,
    cutout_bgr: np.ndarray,
) -> dict:
    cutouts_dir.mkdir(parents=True, exist_ok=True)
    crop_path = cutouts_dir / f"{base_name}_0.jpg"
    mask_path = cutouts_dir / f"{base_name}_0_mask.png"
    cutout_path = cutouts_dir / f"{base_name}_0.png"

    cv2.imwrite(str(crop_path), crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    cv2.imwrite(str(mask_path), mask_class)
    cv2.imwrite(str(cutout_path), cutout_bgr)

    return {"crop_path": str(crop_path), "mask_path": str(mask_path), "cutout_path": str(cutout_path)}


def write_metadata_json(cutouts_dir: Path, base_name: str, metadata: dict) -> Path:
    cutouts_dir.mkdir(parents=True, exist_ok=True)
    path = cutouts_dir / f"{base_name}_0.json"
    path.write_text(json.dumps(metadata, indent=2, default=str))
    return path
