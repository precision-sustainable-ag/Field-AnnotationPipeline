import json
import math
import copy 
import logging
import exifread
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional, Dict, List
from omegaconf import DictConfig

# Configure logging
log = logging.getLogger(__name__)


class MetadataExtractor:
    """
    A class for extracting and handling image metadata.
    """

    def __init__(self, cfg: DictConfig):
        """
        This function initializes the metadata extractor.

        Parameters:
            cfg (DictConfig): Configuration object containing the paths to the data files.
        
        Returns:
            None
        """
        # Output metadata directory
        self.image_metadata_dir = Path(cfg.paths.temp_dir) / "metadata"
        self.csv_path = cfg.paths.merged_tables_permanent
        self.species_info_path = cfg.paths.field_species_info
        self.metadata_version = cfg.metadata_version
        self.df = pd.read_csv(self.csv_path, low_memory=False)
        self.missing_data_notes = [] # list to store missing data notes

        assert not self.df.empty, "Merged data tables CSV is empty."

        with open(self.species_info_path, "r") as file:
            self.species_info = json.load(file)

        self.species_info_common_name_remapped = {v["common_name"]: {"class_id":v["class_id"], "alias": v["alias"]} for k, v in self.species_info['species'].items()}

    def get_class_id(self, image_name: str) -> Optional[str]:
        """
        This function extracts the class ID from the image name.

        Parameters:
            image_name (str): Name of the image.

        Returns:
            str: Class ID of the species if found; otherwise, returns None.
        """
        log.debug(f"Getting class_id for {image_name}.")

        # Find the species for the given image
        species_series = self.df[self.df["Name"].str.lower() == image_name.lower()]["Species"] # Common name
        species = [str(species).lower() for species in species_series][0] 

        # Add logic for finding if species is a Nan
        if species == "nan" or species == np.nan:
            log.error(f"Species data not found for image: {image_name}.")
            self.missing_data_notes.append(f"Missing Species data for {image_name}.")
            return None

        class_id = self._find_class_id(species)
        log.debug(f"Class ID: {class_id}")

        return class_id

    def _find_class_id(self, species: str) -> Optional[str]:
        """
        This function finds the class ID for the given species.

        Parameters:
            species (str): Species name.

        Returns:
            str: Class ID of the species if found; otherwise, returns None.
        """
        species_info_copy = self.species_info['species'] 
        
        # Loop through the species_info dictionary to find the class_id comparing the common_name from the azure table to the species_info dictionary
        for _, values in species_info_copy.items():
            if values["common_name"].lower() == species:
                return values["class_id"]
            elif 'alias' in values: # if different common names exists, use alias to match
                for alias in values['alias']:
                    if alias.lower() == species:
                        return values['class_id']
        
        log.warning(f"Species '{species}' not found in the species info.")
    
    def get_exif_data(self, image_path: str) -> Dict:
        """
        Extracts EXIF metadata from an image and returns it as an ImageMetadata dataclass.

        The function reads the image file, processes the EXIF tags using exifread, 
        formats the metadata into a dictionary, and converts it to an ImageMetadata instance.

        Returns:
            ImageMetadata: The extracted EXIF metadata.
        """
        log.debug("Extracting EXIF data from the image.")

        # Open image file for reading (must be in binary mode)
        f = open(image_path, "rb")
        
        # Return Exif tags
        tags = exifread.process_file(f, details=False)
        f.close()

        # Initialize an empty dictionary to hold processed metadata
        exif_data = {}

        # Iterate over the extracted tags and process them
        for x, y in tags.items():
            # Extract the value, handling lists with a single element
            newval = (
                y.values[0]
                if type(y.values) == list and len(y.values) == 1
                else y.values
            )
            # Convert Ratio objects to strings
            if type(newval) == exifread.utils.Ratio:
                newval = str(newval)

            # Clean up the tag key by removing unnecessary prefixes
            exif_data[x.rsplit(" ")[1]] = newval

        # Filter exif data
        filtered_exif_data = self.filter_exif_data(exif_data)
        return filtered_exif_data
        
    def save_image_metadata(
            self, 
            image_path: Path, 
            image_info: Dict, 
            annotations: Dict, 
            plant_field_info: Dict, 
            category: Dict, 
            exif_data: Dict
            ) -> None:
        """
        This function saves the metadata extracted from the image.
        """
        
        # Add missing data note to the metadata if any data is missing
        if self.missing_data_notes:
            image_info["Note"] = " ".join(self.missing_data_notes)
        else:
            image_info["Note"] = None
            
        # Combine the extracted metadata into a single dictionary
        combined_dict = {
            "image_info": image_info,
            "plant_field_info": plant_field_info,
            "annotation": annotations,
            "category": category,
            "exif_meta": exif_data,
            "version": self.metadata_version
        }

        # Save the metadata to a JSON file
        assert image_path.parent.name == "developed-images", f"Expected image_path.parent.name to be 'developed-images', but got {image_path.parent.name}."
        metadata_dir = image_path.parent.parent / "cutouts"
        metadata_filename = metadata_dir / f"{image_path.stem}_0.json"
        with open(metadata_filename, "w") as file:
            json.dump(combined_dict, file, indent=4, default=str)
        
        log.debug(f"Metadata saved for {metadata_filename.name}.")

    def get_image_info(self, image_info: pd.DataFrame) -> Dict:
        """
        This function extracts the image information from the dataframe.

        Parameters:
            image_info (pd.DataFrame): Dataframe containing the image information.

        Returns:
            dict: Extracted image information.
        """

        # Extract the relevant dataf from the image_info saved on the tablet when the image was taken
        image_info_list = [
            "Name", "Extension", "Batch_id", "ImageURL", "UploadDateTimeUTC", "CameraInfo_DateTime", "SizeMiB", "HasMatchingJpgAndRaw", "ImageIndex", "UsState"
        ]

        camerainfo_date_time = image_info['CameraInfo_DateTime'].iloc[0]
        camerainfo_date = camerainfo_date_time.split(" ")[0]

        batch_id = [f"{image_info['UsState'].iloc[0]}_{camerainfo_date}"]

        image_info.insert(2, "Batch_id", batch_id)

        image_info_imp = image_info[image_info_list].to_dict(orient='list')

        image_info_dict = {key: value[0] if value else None for key, value in image_info_imp.items()}

        image_info_dict["Name"] = image_info_dict["Name"].split(".")[0]

        image_info_dict = self._custom_decoder(image_info_dict) # deal with NaN values and en dash
        image_info_dict = self._replace_en_dash(image_info_dict) # deal with en dash

        return image_info_dict

    def get_plant_field_info(self, image_info: pd.DataFrame) -> Dict:
        """
        This function extracts the plant field information from the dataframe.

        Parameters:
            image_info (pd.DataFrame): Dataframe containing the image information.

        Returns:
            dict: Extracted plant field information.
        """
        # Extract the relevant columns from the plant_field_info
        plant_field_info_list = [
            "PlantType", "CloudCover", "GroundResidue", "GroundCover", "CoverCropFamily", "GrowthStage", "CottonVariety", "CropOrFallow",
            "CropTypeSecondary", "Species", "Height", "SizeClass", "FlowerFruitOrSeeds"
        ]

        plant_field_info = image_info[plant_field_info_list].to_dict(orient='list')
        plant_field_info_dict = {key: value[0] if value else None for key, value in plant_field_info.items()}

        plant_field_info_dict = self._custom_decoder(plant_field_info_dict) # deal with NaN values and en dash
        plant_field_info_dict = self._replace_en_dash(plant_field_info_dict) # deal with en dash

        return plant_field_info_dict

    def get_category(self, image_name: str) -> Dict:
        """
        This function extracts the category information based on the class ID.

        Parameters:
            image_name (str): Name of the image.

        Returns:
            dict: Extracted category information.
        """
        class_id = self.get_class_id(image_name)

        # Make a copy of the species_info dictionary to avoid modifying the original
        species_info_copy = copy.deepcopy(self.species_info['species'])

        for _, species_value in species_info_copy.items():
            if species_value['class_id'] == class_id:
                category = species_value
                # category.pop('alias') if 'alias' in category else None # delete class_id in the final metadata
                break
            elif class_id is None:
                category = None

        if category is not None:
            for key in ('collection_location', 'collection_timing'):
                category.pop(key, None)  # `None` prevents KeyError if key doesn't exist

        return category

    def filter_exif_data(self, exif_data: Dict) -> Dict:
        """
        This function extracts the relevant EXIF data.

        Parameters:
            exif_data (dict): Extracted EXIF data.

        Returns:
            dict: Extracted relevant EXIF data.
        """
        # Extract the relevant EXIF data
        exif_data_list = [
            "ExifImageWidth", "ExifImageLength", "Make", "Model", "Software", "DateTime", "ExposureTime", "FNumber", "ExposureProgram", "ISOSpeedRatings", 
            "RecommendedExposureIndex", "ExifVersion", "BrightnessValue", "MaxApertureValue", "LightSource", "Flash", "FocalLength", "ExposureMode", 
            "WhiteBalance", "FocalLengthIn35mmFilm", "Contrast", "Saturation", "Sharpness", "LensModel", "LensSpecification", "BodySerialNumber"
        ]

        exif_data_imp_dict = {key: exif_data[key] for key in exif_data_list if key in exif_data}

        exif_data_imp_dict = {key.replace("ExifImageWidth", "ImageWidth"): value for key, value in exif_data_imp_dict.items()}
        exif_data_imp_dict = {key.replace("ExifImageLength", "ImageLength"): value for key, value in exif_data_imp_dict.items()}

        exif_data_imp_dict = self._custom_decoder(exif_data_imp_dict) # deal with NaN values and en dash
        exif_data_imp_dict = self._replace_en_dash(exif_data_imp_dict) # deal with en dash

        return exif_data_imp_dict

    

    @staticmethod
    def _custom_decoder(data: Dict) -> Dict:
        """
        This function replaces NaN values with None in the dictionary.

        Parameters:
            data (dict): Dictionary to modify.

        Returns:
            dict: Modified dictionary.
        """
        for key, value in data.items():
            if isinstance(value, float) and math.isnan(value):
                data[key] = None
            elif isinstance(value, dict):
                data[key] = MetadataExtractor._custom_decoder(value)
        return data

    @staticmethod
    def _replace_en_dash(data: Dict) -> Dict:
        """
        This function replaces en dash with hyphen in the dictionary.

        Parameters:
            data (dict): Dictionary to modify.

        Returns:
            dict: Modified dictionary.
        """
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = value.replace('\u2013', '-')
            elif isinstance(value, dict):
                data[key] = MetadataExtractor._replace_en_dash(value)
        return data

class ProcessDetections:
    """
    A class to orchestrate the weed detection and metadata extraction process.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes the ProcessDetections class.

        Parameters:
            cfg (DictConfig): Configuration object containing the paths to the data files.

        Returns:    
            None
        """
        self.output_dir = Path(cfg.paths.temp_dir)
        self.metadata_extractor = MetadataExtractor(cfg)

        self.annotations_template = {
            "bbox": None,
            "det_pred_conf": None,
            "HasMatPred": None, 
            "HasMatPredConf": None
        }

    def process_image_sequentially(self, image_path: Path) -> None:
        """
        This function processes the image by detecting weeds and saving the metadata.

        Parameters:
            image_path (Path): Path to the image file.

        Returns:
            None
        """
        image_name = image_path.name
        # Get the table metadata
        table_metadata = self.metadata_extractor.df[self.metadata_extractor.df["Name"].str.lower() == image_name.lower()]
        
        # Get image info
        image_info = self.metadata_extractor.get_image_info(table_metadata)

        # Get "annotation" metadata
        annotations = self.annotations_template
        
        # Get plant_field_info metadata
        plant_field_info = self.metadata_extractor.get_plant_field_info(table_metadata)

        # category metadata
        category = self.metadata_extractor.get_category(image_name)

        # exif_metadata    
        exif_data = self.metadata_extractor.get_exif_data(str(image_path))

        # Save the image metadata
        self.metadata_extractor.save_image_metadata(
            image_path, 
            image_info,
            annotations, 
            plant_field_info, 
            category, 
            exif_data
            )
    
    
    def process_images(self) -> None:
        """
        This function processes all the images in the directory.

        Returns:
            None
        """
        log.info("Processing images.")
        # Loop through the batches
        batches = list(self.output_dir.iterdir())
        for batch in batches:
            image_dir = Path(batch /"developed-images")
            image_metadata_dir = batch /  'cutouts' # save metadata in the same batch as the image
            image_metadata_dir.mkdir(exist_ok=True)
            # Loop through the images in the batch
            image_paths = sorted(list(image_dir.glob("*.JPG")) + list(image_dir.glob("*.jpg")))
            for image_path in image_paths:
                    self.metadata_extractor.missing_data_notes = [] # reset missing data notes for each image
                    self.process_image_sequentially(image_path)
            log.info(f"Processed {len(image_paths)} images in {batch.name}.")
        log.info("All images processed.")

def main(cfg: DictConfig) -> None:
    """
    Main function to start the weed detection process.

    Parameters:
        cfg (DictConfig): Configuration object containing the paths to the data files.

    Returns:
        None
    """
    log.info(f"Starting {cfg.general.task}")
    processdetections = ProcessDetections(cfg)
    processdetections.process_images()
    log.info(f"{cfg.general.task} completed.")
