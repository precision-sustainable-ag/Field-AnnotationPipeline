import shutil
import pandas as pd
import logging
from pathlib import Path
from random import sample

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AquireRandomImagesBySpecies:
    """
    Class to acquire and organize a random selection of images for specific plant species 
    from a large dataset for the purpose of annotation (e.g., in CVAT).
    """
    def __init__(self) -> None:
        """
        Initializes paths to datasets and directories, loads metadata and species information.
        """
        logging.info("Initializing AquireRandomImagesBySpecies class.")
        # Load metadata table with image details
        self.df = pd.read_csv('/home/nsingh27/Field-AnnotationPipeline/data/field-tools/persistent_tables/merged_blobs_tables_metadata_lts.csv', low_memory=False)
        
        # Load species name and alias information
        self.species_info_json = pd.read_json('/home/nsingh27/Field-AnnotationPipeline/data/field-utils/species_info.json')
        
        # Get species list from the JSON data
        self.species_list = self.get_species_list()
        
        # Number of images to sample per species
        self.num_random_images = 20
        
        # Directories
        self.lts_image_dir = Path('/mnt/research-projects/r/raatwell/longterm_images3/')
        self.field_batches_dir = self.lts_image_dir / 'field-batches'
        self.cvat_images_dir = Path('/home/nsingh27/Field-AnnotationPipeline/data_cvat')

    def get_species_list(self):
        """
        Extracts a list of species names and aliases from the species JSON file.

        Returns:
            list: A list of species names and aliases, excluding 'background'.
        """
        logging.info("Extracting species list from the species_info.json.")
        species_list = [
            item['common_name']
            for _, value in self.species_info_json.items()
            for item in value if isinstance(item, dict) and 'common_name' in item
        ]
        species_list.extend([
            alias
            for _, value in self.species_info_json.items()
            for item in value if isinstance(item, dict) and 'alias' in item
            for alias in item['alias']
        ])
        # Exclude 'background' label
        self.species_list = [species for species in species_list if species != 'background']
        return self.species_list

    def get_list_of_developed_images_paths(self):
        """
        Retrieves all image paths from the long-term storage's developed image folders.

        Returns:
            list: List of Path objects pointing to .jpg images.
        """
        developed_images = []
        for folder in self.field_batches_dir.glob('*/'):
            developed_images.extend((folder / 'developed-images').glob('*.jpg'))
        return developed_images

    def get_random_images(self, df_filtered: pd.DataFrame, species: str):
        """
        Selects a random set of image names for the given species, 
        ensuring those images exist in long-term storage.

        Args:
            df_filtered (pd.DataFrame): DataFrame filtered for the species.
            species (str): Name of the species.

        Returns:
            list: Selected image filenames.
        """
        logging.info(f"Selecting {self.num_random_images} random images for species: {species}")
        
        lts_images = self.get_list_of_developed_images_paths()
        lts_image_names = {img.name for img in lts_images}
        
        # Normalize extensions to lowercase
        species_images = [str(name).replace('JPG', 'jpg') for name in df_filtered['Name'].tolist()]
        
        # Only keep images that exist in LTS
        valid_images = [name for name in species_images if name in lts_image_names]
        
        if len(valid_images) > self.num_random_images:
            selected_images = sample(valid_images, self.num_random_images)
        else:
            logging.warning(f"Only {len(valid_images)} images found for species {species}.")
            selected_images = valid_images
        
        return selected_images

    def copy_images(self, image_names: list, species: str, species_dir: Path):
        """
        Copies selected images to the target directory for annotation.

        Args:
            image_names (list): Filenames to be copied.
            species (str): Species name for logging.
            species_dir (Path): Destination directory.
        """
        logging.info(f"Copying {len(image_names)} images for species: {species} to '{species_dir}'.")
        for image_path in self.get_list_of_developed_images_paths():
            if image_path.name in image_names:
                shutil.copy(image_path, species_dir)

    def select_and_copy_species_images(self):
        """
        Iterates through all species, filters the metadata for each species,
        randomly selects valid images, and copies them to a structured output directory.
        """
        logging.info("Starting species image selection and copying process.")
        for species in self.species_list:
            species_dir = self.cvat_images_dir / "species_dirs" / species
            species_dir.mkdir(parents=True, exist_ok=True)

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
    Initializes the class and starts image selection and copying.
    """
    logging.info("Aquiring random images by species.")
    downloader = AquireRandomImagesBySpecies()
    downloader.select_and_copy_species_images()
    logging.info("Images acquired.")

if __name__ == "__main__":
    main()