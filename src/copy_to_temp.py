import logging
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)

class BatchDownloader:
    """Downloads a specific image batch identified by batch_id."""

    def __init__(self, cfg, batch_id: str):
        self.cfg = cfg
        self.batch_id = batch_id
        
        self.src_longterm_developed = Path(cfg.paths.longterm_storage, "field-batches", self.batch_id, "developed-images")
        self.dst_temp_developed = Path(cfg.paths.temp_dir, self.batch_id, "developed-images")

    def get_unprocessed_images(self):
        """Get the list of images that are not in the LTS cutouts folder."""
        src_cutout_dir = self.src_longterm_developed.parent / "cutouts"
        
        src_developed_imgs = list(self.src_longterm_developed.glob("*.jpg"))
        src_developed_stems = [img.stem for img in src_developed_imgs]

        if not src_cutout_dir.exists():
            return src_developed_imgs
        
        src_cutout_stems = [img.stem.replace("_0", "") for img in src_cutout_dir.glob("*.json")]
        # Get the stems that are in the developed folder but not in the cutout folder
        not_in_cutouts = [stem for stem in src_developed_stems if stem not in src_cutout_stems]
        not_in_cutouts = [img for img in src_developed_imgs if img.stem in not_in_cutouts]
        return not_in_cutouts


    def download_image(self, src: Path, dest: Path):
        try:
            shutil.copy2(src, dest)
            log.debug(f"Copied {src} to {dest}")
        except Exception as e:
            log.error(f"Failed to copy {src} to {dest}: {e}")
    
    def check_directories(self):
        """Check if the batch already exists in temp storage and if it does, make sure the number of files match."""
        proceed_flag = False
        if not self.src_longterm_developed.exists():
            log.error(f"Batch {self.batch_id} developed-images folder not found in long-term storage. Exiting.")
            return proceed_flag
            
        else:
            proceed_flag = True
            return proceed_flag
    
    def download_batch(self):
    
        directories_pass = self.check_directories()
        
        if not directories_pass:
            log.error(f"Directory check failed for batch {self.batch_id}. Cannot download from LTS. Exiting.")
            return
        
        src_developed_imgs = self.get_unprocessed_images()
        if not src_developed_imgs:
            log.info(f"All images in batch {self.batch_id} have been processed. Exiting.")
            return
        
        self.dst_temp_developed.mkdir(parents=True, exist_ok=True)
        # Use ThreadPoolExecutor for concurrent file transfers
        with ThreadPoolExecutor(max_workers=12) as executor:  # Adjust max_workers based on available resources
            futures = []
            for file in src_developed_imgs:
                futures.append(executor.submit(self.download_image, file, self.dst_temp_developed / file.name))

            # Wait for all tasks to complete
            for future in as_completed(futures):
                future.result()  # This will raise any exception that occurred in the thread

def main(cfg):
    log.info(f"Starting {cfg.general.task}")
    batch_id = cfg.batch_id
    downloader = BatchDownloader(cfg, batch_id)
    downloader.download_batch()
    log.info(f"Batch {batch_id} downloaded to {downloader.dst_temp_developed}")