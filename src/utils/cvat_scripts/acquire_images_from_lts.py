import shutil
import pandas as pd
import logging
from pathlib import Path
from random import sample

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AcquireRandomImagesBySpecies:
    """
    Class to acquire and organize a random selection of plant species images 
    from a long-term dataset. Useful for tasks like dataset curation and 
    annotation in tools like CVAT.
    """

    def __init__(self, data_dir_path, species_info_json_path, lts_image_dir) -> None:
        """
        Initializes file paths, reads metadata and species information, and sets parameters.

        Args:
            data_dir_path (Path): Path to the root data directory.
            species_info_json_path (Path): Path to the JSON file with species names and aliases.
            lts_image_dir (Path): Path to the long-term storage directory containing images.
        """
        logging.info("Initializing AcquireRandomImagesBySpecies class.")

        self.lts_image_dir = lts_image_dir
        self.field_batches_dir = self.lts_image_dir / "field-batches"
        self.cvat_images_dir = data_dir_path / "data_cvat"
        persistent_table_path = data_dir_path / "field-tools/persistent_tables/merged_blobs_tables_metadata_lts.csv"

        self.df = pd.read_csv(persistent_table_path, low_memory=False)  # Main metadata table
        self.species_info_json = pd.read_json(species_info_json_path)  # Species info (names + aliases)
        self.species_list = self.get_species_list()  # All species to process
        self.num_random_images = 20  # Default number of images per species to sample

    def get_species_list(self):
        """
        Extracts species names and their aliases from the species_info JSON.

        Returns:
            list: List of species common names and aliases, excluding 'background'.
        """
        logging.info("Extracting species list from the species_info.json.")
        
        # Collect all common names
        species_list = [
            item['common_name']
            for _, value in self.species_info_json.items()
            for item in value if isinstance(item, dict) and 'common_name' in item
        ]
        
        # Collect all aliases
        species_list.extend([
            alias
            for _, value in self.species_info_json.items()
            for item in value if isinstance(item, dict) and 'alias' in item
            for alias in item['alias']
        ])
        
        # Remove background class if present
        self.species_list = [species for species in species_list if species != 'background']
        return self.species_list

    def get_list_of_developed_images_paths(self):
        """
        Scans long-term image directory and retrieves paths to all JPG images 
        inside 'developed-images' subfolders.

        Returns:
            list: List of Path objects pointing to image files.
        """
        developed_images = []
        for folder in self.field_batches_dir.glob('*/'):
            developed_images.extend((folder / 'developed-images').glob('*.jpg'))
        return developed_images

    def get_random_images(self, df_filtered: pd.DataFrame, species: str):
        """
        Filters image names from metadata, verifies if the corresponding files exist 
        in long-term storage, and randomly selects a subset.

        Args:
            df_filtered (pd.DataFrame): DataFrame filtered to contain rows of the target species.
            species (str): Name of the species (for logging and filtering).

        Returns:
            list: List of valid image filenames selected randomly.
        """
        logging.info(f"Selecting {self.num_random_images} random images for species: {species}")
        
        # Get available image filenames in LTS
        lts_images = self.get_list_of_developed_images_paths()
        lts_image_names = {img.name for img in lts_images}
        
        # Normalize image names from metadata and filter those that exist in LTS
        species_images = [str(name).replace('JPG', 'jpg') for name in df_filtered['Name'].tolist()]
        valid_images = [name for name in species_images if name in lts_image_names]
        
        # Randomly sample from valid images
        if len(valid_images) > self.num_random_images:
            selected_images = sample(valid_images, self.num_random_images)
        else:
            logging.warning(f"Only {len(valid_images)} images found for species {species}.")
            selected_images = valid_images
        
        return selected_images

    def copy_images(self, image_names: list, species: str, species_dir: Path):
        """
        Copies selected image files to the designated species directory.

        Args:
            image_names (list): List of image filenames to copy.
            species (str): Name of the species (for logging purposes).
            species_dir (Path): Destination directory to store copied images.
        """
        logging.info(f"Copying {len(image_names)} images for species: {species} to '{species_dir}'.")
        for image_path in self.get_list_of_developed_images_paths():
            if image_path.name in image_names:
                shutil.copy(image_path, species_dir)

    def select_and_copy_species_images(self):
        """
        For each species in the list:
        - Filter metadata to get image records for that species.
        - Randomly select a set of valid image filenames.
        - Copy those images to a species-specific folder under CVAT directory.
        """
        logging.info("Starting species image selection and copying process.")
        
        for species in self.species_list:
            species_dir = self.cvat_images_dir / "species_dirs" / species
            species_dir.mkdir(parents=True, exist_ok=True)

            # Filter dataframe to current species and jpg images only
            df_filtered = self.df[self.df['Species'].str.lower() == species.lower()]
            df_filtered = df_filtered[df_filtered['Extension'] == 'jpg']

            if not df_filtered.empty:
                random_images = self.get_random_images(df_filtered, species)
                self.copy_images(random_images, species, species_dir)
            else:
                logging.warning(f"No images found for species: {species}")

def main():
    """
    Entrypoint for script execution. 
    Instantiates the image acquirer class and triggers the image selection workflow.
    """
    logging.info("Acquiring random images by species.")
    
    data_dir_path = Path("/home/nsingh27/Field-AnnotationPipeline/data")
    species_info_json_path = Path("/home/nsingh27/Field-AnnotationPipeline/data/field-utils/species_info.json")
    lts_image_dir = Path('/mnt/research-projects/r/raatwell/longterm_images3/')

    image_acquirer = AcquireRandomImagesBySpecies(data_dir_path, species_info_json_path, lts_image_dir)
    image_acquirer.select_and_copy_species_images()
    
    logging.info("Images acquired.")

if __name__ == "__main__":
    main()
