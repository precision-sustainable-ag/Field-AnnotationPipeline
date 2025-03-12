"""
This script ensures that local model and metadata files are up-to-date with their remote counterparts. 
It compares files based on their SHA-256 hash. If a local file is missing or differs from the remote 
version, it is automatically downloaded and updated.
"""

import logging
import hashlib
import shutil
from pathlib import Path
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def compute_file_hash(filepath: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash for a file."""
    hash_func = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def files_are_identical(local_file: Path, remote_file: Path) -> bool:
    """Compare hash of local and remote files to determine if they are identical."""
    if not local_file.exists() or not remote_file.exists():
        log.warning(f"Missing file: {local_file} or {remote_file}")
        return False

    local_hash = compute_file_hash(local_file)
    remote_hash = compute_file_hash(remote_file)
    
    return local_hash == remote_hash

def update_file(local_file: Path, remote_file: Path):
    """Copy remote file to local if they are different."""
    try:
        log.info(f"Updating {local_file} from {remote_file}")
        shutil.copy2(remote_file, local_file)
        log.info(f"Update complete: {local_file}")
    except Exception as e:
        log.error(f"Failed to update {local_file}: {e}")

def main(cfg: DictConfig) -> None:
    """
    Checks if local files are up to date compared to their remote versions,
    and updates them if necessary.
    """
    model_files = {
        "classifier_model": {
            "local": Path(cfg.paths.yolo_mat_classifier),
            "remote": Path("/mnt/research-projects/r/raatwell/longterm_images3/field-tools/models/yolo_mat_classifier/train2/weights/best.pt"),
        },
        "detection_model": {
            "local": Path(cfg.paths.yolo_weed_detection_model),
            "remote": Path("/mnt/research-projects/r/raatwell/longterm_images3/field-tools/models/yolo_weed_detection/weights/best.pt"),
        },
        "segmentation_model": {
            "local": Path(cfg.paths.unet_segmentation_model),
            "remote": Path("/mnt/research-projects/r/raatwell/longterm_images3/field-tools/models/unet_segmentation_model/weights/unet_segmentation.pth"),
        },
        "species_info": {
            "local": Path(cfg.paths.field_species_info),
            "remote": Path(cfg.paths.longterm_images2) / "semifield-utils" / "species_information" / "species_info.json",
        },
        "persistent_data_table": {
            "local": Path(cfg.paths.merged_tables_permanent),
            "remote": Path("/mnt/research-projects/r/raatwell/longterm_images3/field-tools/persistent_data_tables/merged_blobs_tables_metadata_lts.csv"),
        },
    }

    for name, paths in model_files.items():
        local = paths["local"]
        remote = paths["remote"]

        if not remote.exists():
            log.warning(f"Remote file {remote} does not exist, skipping.")
            continue
        
        if not local.exists():
            log.warning(f"Local file {local} does not exist. Downloading...")
            update_file(local, remote)
            continue
        
        if files_are_identical(local, remote):
            log.info(f"{name}: Local file is up-to-date.")
        else:
            log.warning(f"Local {name} is outdated or different from the remote version. Updating...")
            update_file(local, remote)
