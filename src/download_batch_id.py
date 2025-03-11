import logging
import shutil
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

log = logging.getLogger(__name__)

class BatchDownloader:
    """Downloads a specific image batch identified by batch_id."""

    def __init__(self, cfg, batch_id: str):
        self.cfg = cfg
        self.batch_id = batch_id
        self.longterm_storage = Path(cfg.paths.longterm_storage, "field-batches")
        self.temp_storage = Path(cfg.paths.temp_dir)

    def download_image(self, src: Path, dest: Path):
        try:
            shutil.copy2(src, dest)
            log.debug(f"Copied {src} to {dest}")
        except Exception as e:
            log.error(f"Failed to copy {src} to {dest}: {e}")

    def download_batch(self):
        src_batch_path = self.longterm_storage / self.batch_id
        dst_batch_path = self.temp_storage / self.batch_id

        if not src_batch_path.exists():
            log.error(f"Batch {self.batch_id} not found in long-term storage.")
            sys.exit(1)

        dst_batch_path.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=16) as executor:
            futures = []
            for item in src_batch_path.iterdir():
                if item.name != "raws":
                    dst_path = dst_batch_path / item.name
                    if item.is_dir():
                        dst_path.mkdir(exist_ok=True, parents=True)
                        for file in item.iterdir():
                            futures.append(executor.submit(self.download_image, file, dst_path / file.name))
                    else:
                        futures.append(executor.submit(self.download_image, item, dst_path))

            for future in as_completed(futures):
                future.result()


def main(cfg):
    batch_id = cfg.batch_id
    downloader = BatchDownloader(cfg, batch_id)
    downloader.download_batch()
    log.info(f"Batch {batch_id} downloaded to {downloader.temp_storage}")