import logging
import shutil
from pathlib import Path
import pandas as pd
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)

class BatchDownloader:
    """Downloads a specific image batch identified by batch_id."""

    def __init__(self, cfg, batch_id: str):
        self.cfg = cfg
        self.batch_id = batch_id
        self.plant_type = cfg.plant_type.upper()
        self.metadata_table = self._load_metadata(Path(cfg.paths.merged_tables_permanent))
        
        self.src_longterm_developed = Path(cfg.paths.longterm_storage, "field-batches", self.batch_id, "developed-images")
        self.src_longterm_cutouts = Path(cfg.paths.longterm_storage, "field-batches", self.batch_id, "cutouts")
        self.dst_temp_developed = Path(cfg.paths.temp_dir, self.batch_id, "developed-images")

    def _load_metadata(self, path: Path) -> pd.DataFrame:
        """Load metadata from a CSV file."""
        if path.exists():
            log.debug(f"Loading metadata from {path}")
            df = pd.read_csv(path, low_memory=False)
            
            # Make sure user-specified plant type is in the metadata
            unique_plant_types = df["PlantType"].unique()
            if self.plant_type not in unique_plant_types:
                log.warning(f"Plant type {self.plant_type} not found in metadata. Available types: {unique_plant_types}")
                raise ValueError(f"Plant type {self.plant_type} not found in metadata. Available types: {unique_plant_types}")
            
            # Remove any rows with nan in the PlantType column
            df = df.dropna(subset=["PlantType"])
            # If self.plant_type in df["PlantType"].unique(), filter the dataframe using self.plant_type
            df = df[df["PlantType"].str.contains(self.plant_type, case=False)]
            return df
        else:
            log.error(f"Metadata file {path} does not exist.")
            raise FileNotFoundError(f"Metadata file {path} does not exist.")

    def get_unprocessed_images(self):
        """Get the list of images that are not in the LTS cutouts folder."""        
        # Get the list of images in the LTS developed folder
        
        src_developed_imgs = list(self.src_longterm_developed.glob("*.jpg"))
        src_developed_stems = [img.stem for img in src_developed_imgs]
        
        if not self.src_longterm_cutouts.exists():
            return src_developed_imgs
        
        # Get the list of images in the LTS cutouts folder
        src_cutout_stems = [img.stem.replace("_0", "") for img in self.src_longterm_cutouts.glob("*.json")]
        # Get the stems that are in the developed folder but not in the cutout folder
        not_in_cutouts = [stem for stem in src_developed_stems if stem not in src_cutout_stems]
        not_in_cutouts = [img for img in src_developed_imgs if img.stem in not_in_cutouts]
        return not_in_cutouts
    
    def _filter_by_plant_type(self, images: List) -> List:
        """Filter images by plant type."""
        basename_set = set(self.metadata_table["BaseName"].astype(str))
        filtered_paths = [Path(p) for p in images if Path(p).stem in basename_set]
        return filtered_paths

    def download_image(self, src: Path, dest: Path):
        try:
            shutil.copy2(src, dest)
            log.debug(f"Copied {src} to {dest}")
        except Exception as e:
            log.error(f"Failed to copy {src} to {dest}: {e}")
    
    def check_src_directory(self):
        """Check if the source directory exists."""
        proceed_flag = False
        if not self.src_longterm_developed.exists():
            log.error(f"Batch {self.batch_id} developed-images folder not found in long-term storage. Exiting.")
            return proceed_flag
            
        else:
            proceed_flag = True
            return proceed_flag
    
    def download_batch(self):
        
        # Check if the source directory exists
        directories_pass = self.check_src_directory()
        
        if not directories_pass:
            log.error(f"Directory check failed for batch {self.batch_id}. Cannot download from LTS. Exiting.")
            raise FileNotFoundError(f"Batch {self.batch_id} developed-images folder not found in long-term storage. Exiting.")
        
        # Get unprocessed images from the long-term storage
        src_developed_imgs = self.get_unprocessed_images()

        if not src_developed_imgs:
            log.info(f"All images in batch {self.batch_id} have been processed. Exiting.")
            return
        
        # Filter images by plant type
        filtered_src_developed_imgs = self._filter_by_plant_type(src_developed_imgs)
        
        if not filtered_src_developed_imgs:
            log.info(f"No images found for batch {self.batch_id} using PlantType '{self.plant_type}' in metadata table.")
            raise ValueError(f"No images found for batch {self.batch_id} in metadata table.")
        
        # Get the relative path of self.dst_temp_developed
        rel_path = self.dst_temp_developed.relative_to(self.cfg.paths.workdir)
        
        log.info(f"Downloading {len(filtered_src_developed_imgs)} images from batch {self.batch_id} to {rel_path}")

        self.dst_temp_developed.mkdir(parents=True, exist_ok=True)
        # Use ThreadPoolExecutor for concurrent file transfers
        with ThreadPoolExecutor(max_workers=12) as executor:  # Adjust max_workers based on available resources
            futures = []
            for file in filtered_src_developed_imgs:
                futures.append(executor.submit(self.download_image, file, self.dst_temp_developed / file.name))

            # Wait for all tasks to complete
            for future in as_completed(futures):
                future.result()  # This will raise any exception that occurred in the thread

def main(cfg):
    log.info(f"Starting {cfg.general.task}")
    batch_id = cfg.batch_id
    downloader = BatchDownloader(cfg, batch_id)
    downloader.download_batch()
    log.info(f"Batch {batch_id} downloaded to {downloader.dst_temp_developed.relative_to(cfg.paths.workdir)}")