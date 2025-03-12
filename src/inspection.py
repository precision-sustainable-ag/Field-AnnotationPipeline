import cv2
import random
import logging
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig
from matplotlib import pyplot as plt

# Configure logging
log = logging.getLogger(__name__)

class InspectMetadataCutouts:
    """
    Class for inspecting image cutouts with corresponding segmentation masks.

    """
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the InspectMetadataCutouts class.

        Args:
            cfg (DictConfig): Configuration object containing necessary paths.
        """
        self.cfg = cfg
        self.batch_id = cfg.batch_id
        self.df = pd.read_csv(cfg.paths.merged_tables_permanent, low_memory=False)

        # Number of random images to inspect per batch
        self.num_random_images_to_inspect = cfg.inspection.num_random_images_to_inspect
        
        # Set up directories
        self.temp_dir = Path(cfg.paths.temp_dir)
        self.inspection_dir = Path(self.temp_dir) / self.batch_id / "inspection"
        self.inspection_dir.mkdir(parents=True, exist_ok=True)

    def _read_image_convert_rgb(self, image_path: Path) -> np.ndarray:
        """
        Reads an image using OpenCV and converts it from BGR to RGB.

        Args:
            image_path (Path): Path to the image file.

        Returns:
            np.ndarray: The image in RGB format.
        """
        return cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

    def _save_image(self, inspection_batch: Path, image_path: Path, 
                    cropped_image: np.ndarray, mask: np.ndarray, 
                    cutout: np.ndarray, species: str) -> None:
        """
        Saves an inspection image showing the cropped image, segmentation mask, 
        and cutout image.

        Args:
            inspection_batch (Path): Directory where the inspection images will be saved.
            image_path (Path): Path to the original image.
            cropped_image (np.ndarray): The cropped image array.
            mask (np.ndarray): The segmentation mask array.
            cutout (np.ndarray): The cutout image array.
            species (str): Species label extracted from metadata.
        """
        log.info(f"Saving inspection image for species: {species}")
        image_name = Path(image_path).name
        image_save_path = Path(inspection_batch / image_name)
        _, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        axs[0].imshow(cropped_image)
        axs[0].set_title('Cropped Image')
        axs[0].axis('off')

        axs[1].imshow(mask, cmap='gray', vmin=0, vmax=np.unique(mask)[1]) 
        axs[1].set_title('Mask')
        axs[1].axis('off')
        
        axs[2].imshow(cutout)
        axs[2].set_title('Cutout Image')
        axs[2].axis('off')
        
        plt.suptitle(f'Species: {species}', fontsize=16)
        plt.tight_layout()
        plt.savefig(image_save_path, bbox_inches='tight')
        plt.close()
        log.info(f"Saved inspection image to {image_save_path}")
    
    def _extract_species(self, image_stem: str) -> str:
        """
        Extracts the species label from the metadata dataframe based on the image name.

        Args:
            image_stem (str): The stem (filename without extension) of the image.

        Returns:
            str: The species name associated with the image.
        """
        log.info(f"Extracting species for image: {image_stem}")
        df_stem_jpg = self.df[(self.df['Stem'] == image_stem) & (self.df['Extension'] == 'jpg')]
        species = df_stem_jpg['Species'].values[0]
        return species

    def process_directory(self) -> None:
        """
        Processes all image cutouts within the temporary directory, randomly selecting 
        up to 10 images per batch for inspection, and saving their visualization.
        """
        log.info("Starting inspection image saving process.")
        batch = Path(self.temp_dir / self.batch_id)
        img_dir = Path(batch / "cutouts")
        log.info(f"Inspecting cutouts in batch: {img_dir}")

        inspection_batch = self.inspection_dir # create inspection batch directory

        cropped_images = list(Path(img_dir).rglob("*.jpg"))

        if len(cropped_images) == 0: 
            log.info(f"No processed images found in {batch}.")
        elif 0 < len(cropped_images) < self.num_random_images_to_inspect:
            log.info(f"Found less than {self.num_random_images_to_inspect} images in {batch}. Using all images for inspection.")
            for image_path in cropped_images:
                mask_path  = f"{str(image_path).replace('.jpg', '_mask.png')}"
                cutout_path = f"{str(image_path).replace('.jpg', '.png')}"
                log.info(f"Processing image: {image_path}")
                image_stem = (image_path.stem).replace('_0', '')
                species = self._extract_species(image_stem)
                cropped_image = self._read_image_convert_rgb(image_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                cutout = self._read_image_convert_rgb(cutout_path)
                self._save_image(inspection_batch, image_path, cropped_image, mask, cutout, species)
        else:
            log.info(f"Found more than {self.num_random_images_to_inspect} images in {batch}. Using {self.num_random_images_to_inspect} random images for inspection.")
            randomly_selected_images = random.sample(cropped_images, self.num_random_images_to_inspect)
            for image_path in randomly_selected_images:
                mask_path  = f"{str(image_path).replace('.jpg', '_mask.png')}"
                cutout_path = f"{str(image_path).replace('.jpg', '.png')}"
                log.info(f"Processing image: {image_path}")
                image_stem = (image_path.stem).replace('_0', '')
                species = self._extract_species(image_stem)
                cropped_image = self._read_image_convert_rgb(image_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                cutout = self._read_image_convert_rgb(cutout_path)
                self._save_image(inspection_batch, image_path, cropped_image, mask, cutout, species)
        log.info("Inspection completed.")

def main(cfg: DictConfig) -> None:
    """
    Main function to initialize and start the inspection process.

    Args:
        cfg (DictConfig): Configuration object containing paths and other parameters.
    """
    log.info(f"Starting image inspection process.")
    trainer = InspectMetadataCutouts(cfg)
    trainer.process_directory()
