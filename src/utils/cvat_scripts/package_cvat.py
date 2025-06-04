import cv2
import shutil
import logging
from glob import glob
import numpy as np
from PIL import Image
from pathlib import Path
from random import random

# Configure logging to show timestamps, log level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PackageCVAT:
    """
    Class for packaging image and mask datasets for different species into a format compatible with CVAT annotation tool.

    This class:
    - Organizes images and corresponding binary masks
    - Converts grayscale masks to 3-channel (3D) masks
    - Creates necessary mapping and color label files for CVAT
    - Zips the final directory and removes temporary files
    """
    def __init__(self, species_dir: Path) -> None:
        """
        Initializes the PackageCVAT class by setting up directories and finding input files.

        Args:
            species_dir (Path): Path to the root directory containing a 'cutouts' folder with .jpg images and *_mask.png masks.
        """
        self.species_dir = Path(species_dir)
        self.species = str(self.species_dir.name).replace(" ", "_").lower()
        self.cutouts_dir = self.species_dir / "cutouts"
        
        # Collect all image and mask paths
        self.cropout_images = self.cutouts_dir.glob('*.jpg')
        self.cropout_masks = self.cutouts_dir.glob('*_mask.png')

        # Output CVAT-compatible directory structure
        self.cvat_dir = self.species_dir / f"{self.species}_cvat_package"
        self.cvat_img_dir = self.cvat_dir / f"{self.species}" # For input images
        self.cvat_mask_dir = self.cvat_dir / f"{self.species}annot" # For corresponding masks

        # Create directories if they don't exist
        self.cvat_img_dir.mkdir(parents=True, exist_ok=True)
        self.cvat_mask_dir.mkdir(parents=True, exist_ok=True)

        # Paths for annotation and color-label text files
        self.images_annot_file_path = self.cvat_dir / f"{self.species}.txt"
        self.colorlabel_file_path = self.cvat_dir / "label_colors.txt"

    def process_images_and_masks(self) -> None:
        """
        Processes all images and masks:
        - Creates 3D masks and saves them
        - Writes annotation and label files
        - Zips the entire CVAT package directory and deletes the temp folder
        """
        logging.info(f"Processing images and masks for species: {self.species}")
        image_paths = list(self.cropout_images)
        if not image_paths:
            logging.warning("No images found. Skipping CVAT package creation.")
            return

        for image_path in image_paths:
            # Convert image to .png and copy to CVAT directory
            new_image_path = self.cvat_img_dir / image_path.name
            shutil.copy(str(image_path), new_image_path)

            # Write line in txt file linking image and mask
            with open(self.images_annot_file_path, 'a') as f:
                f.write(f"/{self.species}/{image_path.name} {self.species}annot/{image_path.stem}.png\n")

            # Convert grayscale mask to 3D and save
            mask_3d = self.create_3d_masks(image_path)

        # Write label_colors.txt with RGB and class label
        self.create_colorlabel_file(mask_3d)

        # Zip the CVAT package and clean up
        zip_file_path = self.cvat_dir.with_suffix('.zip')
        shutil.make_archive(str(zip_file_path.with_suffix('')), 'zip', str(self.cvat_dir))
        shutil.rmtree(self.cvat_dir)
        logging.info(f"{self.species} CVAT package created at: {zip_file_path}")

    def create_colorlabel_file(self, mask_3d: np.ndarray) -> None:
        """
        Creates 'label_colors.txt' which defines the color associated with the weed class.
        It uses the second unique value in the mask.

        Args:
            mask_3d (np.ndarray): 3D mask array from which color is extracted.
        """
        mask_color_label = np.unique(mask_3d)[1]  # Use second unique value as label 

        with open(self.colorlabel_file_path, 'w') as f:
            f.write(f"{mask_color_label} {mask_color_label} {mask_color_label} weed\n")

    def create_3d_masks(self, image_path: Path) -> np.ndarray:
        """
        Converts the 1-channel grayscale mask to a 3-channel (3D) mask.

        Args:
            image_path (Path): Path to the input image (used to find corresponding mask).

        Returns:
            np.ndarray: The 3D mask that was saved.
        """
        mask_path = image_path.with_name(image_path.name.replace('.jpg', '_mask.png'))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Stack the mask into 3 channels (needed by CVAT)
        mask_3d = np.stack((mask,) * 3, axis=-1)

        # Save the 3D mask
        mask_3d_save_path = self.cvat_mask_dir / mask_path.name.replace('_mask.png', '.png')
        cv2.imwrite(str(mask_3d_save_path), mask_3d)

        return mask_3d

if __name__ == "__main__":
    # Path to the dataset containing folders for each species
    species_dataset = Path('/mnt/research-projects/r/raatwell/longterm_images3/field-tools/field_test_dataset/species_dataset')
    
    # Loop through each species folder and process images/masks
    for species_dir in species_dataset.glob('*'):
        if species_dir.is_dir():
            package_cvat = PackageCVAT(species_dir)
            package_cvat.process_images_and_masks()