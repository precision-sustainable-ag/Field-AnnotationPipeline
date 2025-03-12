import logging
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)

class Batch2LTS:
    """Downloads a specific image batch identified by batch_id."""

    def __init__(self, cfg, batch_id: str):
        self.cfg = cfg
        self.batch_id = batch_id
        
        self.src_temp_cutouts = Path(cfg.paths.temp_dir) / self.batch_id / "cutouts"
        self.dst_longterm_cutouts = Path(cfg.paths.longterm_storage) / "field-batches" / self.batch_id / "cutouts"
        
        self.src_temp_inspection = Path(cfg.paths.temp_dir) / self.batch_id / "inspection"
        self.dst_longterm_inspection = Path(cfg.paths.longterm_storage) / "field-batches" / self.batch_id / "inspection"

    def upload_file(self, src: Path, dest: Path):
        try:
            shutil.copy2(src, dest)
            log.debug(f"Copied {src} to {dest}")
        except Exception as e:
            log.error(f"Failed to copy {src} to {dest}: {e}")
    
    def check_src_dir(self):
        """Check if the local batch has all the appropriate files."""
        proceed_flag = False
        # check if the temp src directory exists
        if not self.src_temp_cutouts.exists():
            log.error(f"Batch {self.batch_id} cutouts folder not found in local temp. Exiting.")
            return proceed_flag
        
        # check if each file in the batch has a corresponding .json file
        developed_images = self.dst_longterm_cutouts.parent / "developed-images"
        dev_imgs = developed_images.glob("*.jpg")
        dev_stems = [img.stem for img in dev_imgs]
        for dev_stem in dev_stems:
            # Only check for the .json file
            if not (self.src_temp_cutouts / f"{dev_stem}_0.json").exists():
                log.error(f"File {dev_stem}.json not found in batch {self.batch_id}. Cannot continue. Exiting.")
                return proceed_flag
            
        # Check if the inspection has been performed
        if not self.src_temp_inspection.exists():
            log.error(f"Inspection folder not found in batch {self.batch_id}. Cannot continue. Exiting.")
            return proceed_flag
        
        proceed_flag = True
        return proceed_flag
    
    def check_dst_dir(self):
        """Check if the batch has already been uploaded to LTS and if the LTS dst location exists."""

        proceed_flag = False
        if self.dst_longterm_cutouts.exists():
            
            # Get the count of the files in the batch and make sure they match with what's in the local src folder
            src_count = list(self.src_temp_cutouts.glob("*"))
            dst_count = list(self.dst_longterm_cutouts.glob("*"))
            
            if len(dst_count) != len(src_count):
                log.info(f"Batch {self.batch_id} already has a cutout folder. This may be from other plant Types of image that were previously uploaded and processed. Checking for file uniqueness.")
                src_stems = [x.stem for x in src_count]
                dst_stems = [x.stem for x in dst_count]
                if len(set(src_stems).intersection(set(dst_stems))) == 0:
                    proceed_flag = True
                    return proceed_flag
                
                elif set(src_stems) == set(dst_stems):
                    log.info(f"Batch {self.batch_id} already uploaded to long-term storage. Exiting.")
                    return proceed_flag
                
            else:
                log.info(f"Batch {self.batch_id} already uploaded to long-term storage. Exiting.")
                return proceed_flag
        else:
            proceed_flag = True
            return proceed_flag
    
    
    def upload_batch(self):
        """Uploads the batch files using multithreading."""
        src_checked = self.check_src_dir()
        dst_checked = self.check_dst_dir()

        if not src_checked:
            log.error(f"Source check failed for batch {self.batch_id}. Cannot upload to LTS. Exiting.")
            return

        if not dst_checked:
            log.error(f"Destination check failed for batch {self.batch_id}. Cannot upload to LTS. Exiting.")
            return

        # Collect files
        src_meta = list(self.src_temp_cutouts.glob("*.json"))
        src_cutouts = [x for x in self.src_temp_cutouts.glob("*.png") if "_mask" not in x.name]
        src_cropouts = list(self.src_temp_cutouts.glob("*.jpg"))
        src_masks = list(self.src_temp_cutouts.glob("*_mask.png"))
        src_temp_cutout_files = sorted(src_meta + src_cutouts + src_cropouts + src_masks)
        src_temp_inspection_files = sorted(self.src_temp_inspection.glob("*.jpg"))

        log.info(f"Uploading {len(src_temp_cutout_files)} cutout files and {len(src_temp_inspection_files)} inspection files to long-term storage")

        # Ensure destination directories exist
        self.dst_longterm_cutouts.mkdir(parents=True, exist_ok=True)
        self.dst_longterm_inspection.mkdir(parents=True, exist_ok=True)

        # Use ThreadPoolExecutor for concurrent file transfers
        with ThreadPoolExecutor(max_workers=12) as executor:  # Adjust max_workers based on available resources
            futures = []
            for file in src_temp_cutout_files:
                futures.append(executor.submit(self.upload_file, file, self.dst_longterm_cutouts / file.name))

            for file in src_temp_inspection_files:
                futures.append(executor.submit(self.upload_file, file, self.dst_longterm_inspection / file.name))

            # Wait for all tasks to complete
            for future in as_completed(futures):
                future.result()  # This will raise any exception that occurred in the thread

        log.info(f"Batch {self.batch_id} upload complete.")

def main(cfg):
    log.info(f"Starting {cfg.general.task}")
    batch_id = cfg.batch_id
    downloader = Batch2LTS(cfg, batch_id)
    downloader.upload_batch()
    log.info(f"Batch {batch_id} upload to {downloader.dst_longterm_cutouts} complete.")
    log.info(f"Task: {cfg.general.task} completed.")