from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def stage_batch_images(rows: list[dict], local_dir: Path) -> list[dict]:
    """Copies each row's `jpg_path` (from LTS) into
    `local_dir/developed-images/<base_name>.jpg`. Returns the rows that were
    actually staged, each with a `local_jpg_path` key added; a row whose
    source file is missing on LTS is logged and dropped rather than raising,
    so one missing file can't abort a whole batch.
    """
    dest_dir = local_dir / "developed-images"
    dest_dir.mkdir(parents=True, exist_ok=True)

    staged = []
    for row in rows:
        source = Path(row["jpg_path"])
        if not source.exists():
            log.warning("Skipping %s, source not found on LTS: %s", row["base_name"], source)
            continue
        dest = dest_dir / f"{row['base_name']}.jpg"
        shutil.copy2(source, dest)
        staged.append({**row, "local_jpg_path": str(dest)})

    log.info("Staged %d/%d images into %s", len(staged), len(rows), dest_dir)
    return staged


def stage_cutouts_out(local_cutouts_dir: Path, lts_cutouts_dir: Path) -> int:
    if not local_cutouts_dir.exists():
        return 0
    files = [f for f in local_cutouts_dir.iterdir() if f.is_file()]
    if not files:
        return 0
    lts_cutouts_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, lts_cutouts_dir / f.name)
    log.info("Copied %d cutout files to %s", len(files), lts_cutouts_dir)
    return len(files)


def cleanup_local_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
