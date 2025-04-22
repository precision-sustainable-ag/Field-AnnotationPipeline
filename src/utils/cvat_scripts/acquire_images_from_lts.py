import shutil
import pandas as pd
import logging
from pathlib import Path
from random import sample

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AcquireRandomImagesBySpecies:
    """
    A class to select and organize random images of specific plant species from the lts directory dataset. 
    The selected images can be used for labeling or training datasets in 
    annotation tools like CVAT.
    """
    def __init__(self, data_dir_path, lts_image_dir) -> None:
        """
        Initializes necessary paths, reads metadata and species information, and sets parameters.

        Args:
            data_dir_path (Path): Root path to data directory containing persistent table and species_info.json file.
            lts_image_dir (Path): Path to the long-term storage directory containing field images.
        """
        logging.info("Initializing AcquireRandomImagesBySpecies class.")
        self.lts_image_dir = lts_image_dir
        self.field_batches_dir = lts_image_dir / "field-batches"
        self.cvat_images_dir = data_dir_path / "data_cvat"

        self.df = pd.read_csv(Path(data_dir_path / "field-tools/persistent_tables/merged_blobs_tables_metadata_lts.csv"), low_memory=False)
        self.species_info_json = pd.read_json(data_dir_path / "field-utils/species_info.json")
        self.num_random_images = 20  # Number of images to select per species

    def select_and_copy_species_images(self):
        """
        For each species:
        - Skips if the species has already been processed.
        - Filters the metadata for the species.
        - Randomly selects a defined number of images from available matches.
        - Copies selected images to a species-specific folder.
        """
        logging.info("Starting species image selection and copying process.")

        lts_field_batches_cvat = self.lts_image_dir /"field-tools/field_test_dataset/species_dataset"
        processed_field_batches_cvat_list = [folder.name for folder in lts_field_batches_cvat.glob('*/')] 

        species_list = self.get_species_list()

        for species in species_list:
            if str(species).lower() in str(processed_field_batches_cvat_list).lower():
                logging.info(f"Species '{species}' already processed. Skipping.")
            else:
                logging.info(f"Selecting and copying random images for species: {species}")
                df_filtered = self.df[self.df['Species'].str.lower() == species.lower()]
                df_filtered = df_filtered[df_filtered['Extension'] == 'jpg']
                
                random_images_names = self.get_random_images_names(df_filtered, species)
                print(f"Species: {species}, len of random images: {len(random_images_names)}")
                
                if len(random_images_names) != 0:
                    species_dir = self.cvat_images_dir / "species_dirs" / species
                    species_dir.mkdir(parents=True, exist_ok=True)

                    for image_path in self.get_list_of_developed_images_paths():
                        if image_path.name in random_images_names:
                            shutil.copy(image_path, species_dir)

    def get_species_list(self):
        """
        Extracts a list of species common names and aliases from the species_info.json file.

        Returns:
            list: Combined list of species names and their aliases (excluding 'background').
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
        species_list = [species for species in species_list if species != 'background']
        return species_list

    def get_list_of_developed_images_paths(self):
        """
        Retrieves paths to all JPG images located within the 'developed-images' subdirectories 
        of each field batch in the long-term image directory.

        Returns:
            list: List of full Path objects to each JPG image found.
        """
        developed_images = []
        for folder in self.field_batches_dir.glob('*/'):
            developed_images.extend((folder / 'developed-images').glob('*.jpg'))
        return developed_images

    def get_random_images_names(self, df_filtered: pd.DataFrame, species: str):
        """
        From filtered metadata, selects a list of valid image filenames that exist in 
        the long-term storage. Then randomly samples the desired number.

        Args:
            df_filtered (pd.DataFrame): Subset of metadata for a specific species.
            species (str): Current species name for logging.

        Returns:
            list: Randomly selected image filenames that exist in LTS.
        """
        logging.info(f"Selecting {self.num_random_images} random images for species: {species}")
        
        lts_images = self.get_list_of_developed_images_paths()
        lts_image_names = [img.name for img in lts_images]

        species_images = [str(name).replace('JPG', 'jpg') for name in df_filtered['Name'].tolist()]
        valid_images = [name for name in species_images if name in lts_image_names]
        
        if len(valid_images) > self.num_random_images:
            selected_images = sample(valid_images, self.num_random_images)
        else:
            logging.warning(f"Only {len(valid_images)} images found for species {species}.")
            selected_images = valid_images      
        return selected_images


def main():
    """
    Entrypoint for running the script. Sets up the paths and executes the 
    image acquisition process by species.
    """
    logging.info("Acquiring random images by species.")
    
    data_dir_path = Path("/home/nsingh27/Field-AnnotationPipeline/data")
    lts_image_dir = Path('/mnt/research-projects/r/raatwell/longterm_images3/')

    image_acquirer = AcquireRandomImagesBySpecies(data_dir_path, lts_image_dir)
    image_acquirer.select_and_copy_species_images()
    
    logging.info("Images acquired.")

if __name__ == "__main__":
    main()