import logging
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)

class CleanUpLocalTemp:
    """Downloads a specific image batch identified by batch_id."""

    def __init__(self, cfg, batch_id: str):
        self.cfg = cfg
        self.batch_id = batch_id
        
        self.temp_developed = Path(cfg.paths.temp_dir, self.batch_id, "developed-images")
        self.temp_cutouts = Path(cfg.paths.temp_dir, self.batch_id, "cutouts")
        self.temp_inspected = Path(cfg.paths.temp_dir, self.batch_id, "inspection")

        self.lts_developed = Path(cfg.paths.longterm_storage, "field-batches", self.batch_id, "developed-images")
        self.lts_cutouts = Path(cfg.paths.longterm_storage, "field-batches", self.batch_id, "cutouts")
        self.lts_inspected = Path(cfg.paths.longterm_storage, "field-batches", self.batch_id, "inspection")

    def can_remove_local_dir(self):
        """ Check that all the files in the temp directories are in the LTS directories. """
        temp_developed_imgs = [img for img in self.temp_developed.glob("*.jpg")]
        temp_developed_stems = [img.stem for img in temp_developed_imgs]
        temp_cutout_stems = [img.stem for img in self.temp_cutouts.glob("*.json")]
        temp_inspected_stems = [img.stem for img in self.temp_inspected.glob("*.jpg")]

        lts_developed_imgs = [img for img in self.lts_developed.glob("*.jpg")]
        lts_developed_stems = [img.stem for img in lts_developed_imgs]
        lts_cutout_stems = [img.stem for img in self.lts_cutouts.glob("*.json")]
        lts_inspected_stems = [img.stem for img in self.lts_inspected.glob("*.jpg")]

        # Check that the stems in the temp directories are in the LTS directories. In some cases the LTS might have more stems.
        developed_diff = [stem for stem in temp_developed_stems if stem not in lts_developed_stems]
        cutout_diff = [stem for stem in temp_cutout_stems if stem not in lts_cutout_stems]
        inspected_diff = [stem for stem in temp_inspected_stems if stem not in lts_inspected_stems]

        if developed_diff or cutout_diff or inspected_diff:
            log.error(f"Files in temp directories do not match LTS directories. Cannot remove temp directories.")
            return False
        else:
            return True

        
    def cleanup_temp(self):
        """Remove the batch_id folder from the temp directory."""
        can_remove = self.can_remove_local_dir()
        if not can_remove:
            log.error(f"Cannot remove temp directories for batch {self.batch_id}. Exiting.")
            return
        else:
            try:
                shutil.rmtree(self.temp_developed)
                shutil.rmtree(self.temp_cutouts)
                shutil.rmtree(self.temp_inspected)
                log.info(f"Removed temp directories for batch {self.batch_id}.")
            except Exception as e:
                log.error(f"Failed to remove temp directories for batch {self.batch_id}: {e}")
                return
        

    
def main(cfg):
    log.info(f"Starting {cfg.general.task}")
    batch_id = cfg.batch_id
    cleaner = CleanUpLocalTemp(cfg, batch_id)
    cleaner.cleanup_temp()
    log.info(f"Finished {cfg.general.task}")

    