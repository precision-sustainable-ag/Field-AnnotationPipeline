import os
import shutil
import logging
import pandas as pd
from pathlib import Path
from random import sample

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AcquireRandomImagesBySpecies:
    """
    Class to randomly select and copy species-specific images (either from cutouts or developed images)
    into lts for further processing.
    """
    def __init__(self, data_dir_path: Path, lts_image_dir: Path) -> None:
        """
        Initializes the AcquireRandomImagesBySpecies class.

        Args:
            data_dir_path (Path): Path to the root data directory containing metadata and species info.
            lts_image_dir (Path): Path to the long-term storage directory containing image batches.
        """
        logging.info("Initializing AcquireRandomImagesBySpecies class.")
        self.lts_image_dir = lts_image_dir
        self.field_batches_dir = lts_image_dir / "field-batches"
        self.cvat_images_dir = data_dir_path / "data_cvat"
        self.lts_field_batches_cvat = self.lts_image_dir / "field-tools/field_test_dataset/species_dataset"
        self.df = pd.read_csv(Path(data_dir_path / "field-tools/persistent_tables/merged_blobs_tables_metadata_permanent.csv"), low_memory=False)
        self.species_info_json = pd.read_json(data_dir_path / "field-utils/species_info.json")
        self.num_random_images = 20  # Number of images to be selected per species

    def select_and_copy_species_images(self):
        """
        Selects random images for each species and copies them into a structured directory.
        If enough cropped cutouts exist, cutouts are copied; otherwise, developed images are used.
        """
        logging.info("Starting species image selection and copying process.")

        # Gather available image paths
        list_paths_post_segmentation_cropped_images, list_paths_developed_images_pre_segmentation = self._get_list_of_images_paths()

        # List of species already processed
        processed_field_batches_cvat_list = [folder.name for folder in self.lts_field_batches_cvat.glob('*/')]

        # Extract stems (without extensions) of cutouts and developed images
        list_image_names_with_cutouts = [Path(paths).stem.replace('_0', '') for paths in list_paths_post_segmentation_cropped_images]
        list_image_names_developed_without_cutouts = [Path(paths).stem.replace('_0', '') for paths in list_paths_developed_images_pre_segmentation]

        # List of species from persistent table
        species_list = [species for species in self.df["Species"].unique().tolist() if pd.notna(species)]
        
        # List species that have not yet been packaged
        species_not_packaged = [species for species in species_list if species.lower() not in [processed_species.lower() for processed_species in processed_field_batches_cvat_list]]

        for species in species_not_packaged:
            # Filter dataframe for current species
            filtered_df_species = self.df[self.df["Species"] == species]
            filtered_df_species_jpg = filtered_df_species[filtered_df_species["Extension"] == "jpg"]

            list_image_names_for_species = filtered_df_species_jpg["Stem"].tolist()

            # Check available images
            list_images_species_has_cutouts = [image_name for image_name in list_image_names_for_species if image_name in list_image_names_with_cutouts]
            list_images_species_has_developed_but_no_cutouts = [image_name for image_name in list_image_names_for_species if image_name in list_image_names_developed_without_cutouts and image_name not in list_images_species_has_cutouts]

            if len(list_images_species_has_cutouts) < self.num_random_images:
                logging.info(f"Species '{species}' does not have enough cutouts. Copying developed images instead.")
                
                randomly_selected_developed_images_for_species = sample(
                    list_images_species_has_developed_but_no_cutouts,
                    min(len(list_images_species_has_developed_but_no_cutouts), self.num_random_images)
                )
                
                paths_randomly_selected_developed_images_for_species = [
                    image_path for image_path in list_paths_developed_images_pre_segmentation
                    if image_path.stem.replace('_0', '') in randomly_selected_developed_images_for_species
                ]

                # Copy developed images
                for image_path in paths_randomly_selected_developed_images_for_species:
                    dst_dir = self.lts_field_batches_cvat / species / "developed-images"
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy(image_path, dst_dir)

            else:
                logging.info(f"Species '{species}' has enough cutouts. Copying cutout data.")
                
                randomly_selected_cutouts_for_species = sample(
                    list_images_species_has_cutouts,
                    min(len(list_images_species_has_cutouts), self.num_random_images)
                )
                
                paths_randomly_selected_cropped_images_for_species = [
                    image_path for image_path in list_paths_post_segmentation_cropped_images
                    if image_path.stem.replace('_0', '') in randomly_selected_cutouts_for_species
                ]

                parent_dir_cutouts = paths_randomly_selected_cropped_images_for_species[0].parent

                # Copy cutout-related files
                for file in parent_dir_cutouts.glob("*"):
                    if str(file.name)[:8] in randomly_selected_cutouts_for_species:
                        dst_dir = self.lts_field_batches_cvat / species / "cutouts"
                        os.makedirs(dst_dir, exist_ok=True)
                        shutil.copy(file, dst_dir)

    def _get_list_of_images_paths(self):
        """
        Collects lists of available cutouts (post-segmentation) images and developed images
        from all field batches.

        Returns:
            tuple: (list of cropped images paths, list of developed images paths)
        """
        list_paths_post_segmentation_cutout_images = []
        list_paths_developed_images_pre_segmentation = []

        field_batches = [folder for folder in Path(self.field_batches_dir).glob("*/") if folder.is_dir()]
        for batch in field_batches:
            if (batch / 'cutouts').is_dir():
                list_paths_post_segmentation_cutout_images.extend(
                    path for path in (batch / 'cutouts').glob('*.jpg')
                )
            else:
                list_paths_developed_images_pre_segmentation.extend(
                    path for path in (batch / 'developed-images').glob('*.jpg')
                )

        return list_paths_post_segmentation_cutout_images, list_paths_developed_images_pre_segmentation

def main(data_dir_path: Path, lts_image_dir: Path):
    """
    Main function to initiate the image acquisition process by species.

    Args:
        data_dir_path (Path): Path to the data directory containing metadata and utilities.
        lts_image_dir (Path): Path to the long-term image storage directory.
    """
    logging.info("Acquiring random images by species.")
    
    image_acquirer = AcquireRandomImagesBySpecies(data_dir_path, lts_image_dir)
    image_acquirer.select_and_copy_species_images()
    
    logging.info("Images acquired.")

if __name__ == "__main__":
    data_dir_path = Path("/home/nsingh27/Field-AnnotationPipeline/data")
    lts_image_dir = Path("/mnt/research-projects/r/raatwell/longterm_images3/")
    
    main(data_dir_path, lts_image_dir)
