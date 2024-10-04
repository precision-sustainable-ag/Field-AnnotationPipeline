import json
import logging
import cv2
import numpy as np
import pandas as pd
import math
import exifread
import copy 
import os

from pathlib import Path
from omegaconf import DictConfig
from ultralytics import YOLO
from typing import Optional, Dict

# Configure logging
log = logging.getLogger(__name__)

class ImageLoader:
    """
    A class for loading images from a given directory.
    """
    def __init__(self, image_dir: Path):
        self.image_dir = image_dir

    def read_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        This function reads the image from the given path.
        """
        log.debug(f"Reading image from {image_path}.")
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        return image

class WeedDetector:
    """
    A class for detecting weeds in images using YOLOv5.
    """

    def __init__(self, model_path: str) -> None:
        """
        Initializes the WeedDetector class.
        
        Parameters: 
            model_path (str): Path to the YOLOv5 model.
        """
        self.model = YOLO(model_path)

    def detect_weeds(self, image: np.ndarray) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Detects target weed in the given image.

        Parameters:
            image (np.ndarray): Image as a numpy array.

        Returns:
            dict: Detection results including bbox if detection is successful; otherwise, returns None.
        """
        log.debug("Starting weed detection.")
        results = self.model(image)
        
        if not results or not results[0].boxes.xyxy.tolist():
            log.warning("No detection found.")
            return None

        # Extract the bounding box coordinates
        bbox = results[0].boxes.xyxy.tolist()[0]
        x_min, y_min, x_max, y_max = map(int, bbox)
        detection_results = {
            "image_id": None,
            "bbox": {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            }
        }
        return detection_results


class MetadataExtractor:
    """
    A class for extracting and handling image metadata.
    """

    def __init__(self, cfg: DictConfig):
        """
        This function initializes the metadata extractor.

        Parameters:
            cfg (DictConfig): Configuration object containing the paths to the data files.
        """
        self.csv_path = cfg.data.merged_tables_permanent
        self.species_info_path = cfg.data.field_species_info

        # Load the merged data tables CSV and species info JSON
        self.df = pd.read_csv(self.csv_path, low_memory=False)
        assert not self.df.empty, "Merged data tables CSV is empty."

        with open(self.species_info_path, "r") as file:
            self.species_info = json.load(file)

    def get_class_id(self, image_name: str) -> Optional[str]:
        """
        Extracts the class ID from field_species_info for a given image by
        looking up the species using the persistent data table.

        Parameters:
            image_name (str): Name of the image.

        Returns:
            Optional[str]: Class ID if species is found; otherwise, None.
        """
        log.debug(f"Loading species data for '{image_name}'.")

        # Find species using case-insensitive matching
        species_series = self.df.loc[self.df["Name"].str.lower() == image_name.lower(), "Species"]

        if species_series.empty:
            log.warning(f"Image name not found in the persistent data table for '{image_name}'. Returning None for class ID.")
            return None

        # Drop None/NaN values and grab the first valid species
        species = species_series.dropna().str.lower()

        if species.empty:
            log.warning(f"Species is None in the persistent table or invalid for image: '{image_name}'. Returning class id 27 for 'Unknown plant'.")
            return 28

        species_name = species.iloc[0]  # Get the first non-null species

        # Get the class ID based on species
        class_id = self._find_class_id(species_name)

        if not class_id:
            log.warning(f"Class ID not found for species: '{species}'")
            return None

        log.info(f"Class ID for image '{image_name}' is '{class_id}'.")
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
                if values['alias'].lower() == species:
                    return values['class_id']
        
        log.error(f"Species '{species}' not found in the species info. Returning None.")
        return None
    
    @staticmethod
    def get_exif_data(image_path: str) -> dict:
        """
        Extracts EXIF metadata from an image and returns it as an ImageMetadata dataclass.
        """
        log.info("Extracting EXIF data from the image.")

        # Open image file for reading (must be in binary mode)
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        
        # Initialize an empty dictionary to hold processed metadata
        filtered_exif_data = {}

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
            filtered_exif_data[x.rsplit(" ")[1]] = newval

        return filtered_exif_data
        
    def save_image_metadata(self, image_path: str, detection_results: dict, image_metadata_dir: Path, exif_data: dict) -> None:
        """
        This function saves the metadata extracted from the image.
        """

        log.debug("Extracting metadata")

        # Extract the image information, plant field information, category, and relevant EXIF data
        image_name = Path(image_path).name
        image_info = self.df[self.df["Name"].str.lower() == image_name.lower()]
        if image_info.empty:
            raise ValueError(f"Image '{image_name}' not found in the dataframe.")

        image_info_dict = self._get_image_info(image_info)
        plant_field_info_dict = self._get_plant_field_info(image_info)
        category = self._get_category(image_name)
        exif_data_imp_dict = self._get_exif_data(exif_data)

        # Combine the extracted metadata into a single dictionary
        combined_dict = {
            "image_info": image_info_dict,
            "plant_field_info": plant_field_info_dict,
            "annotation": self._get_bbox_xywh(detection_results),
            "category": category,
            "exif_meta": exif_data_imp_dict
        }

        # Save the metadata to a JSON file
        metadata_filename = image_metadata_dir / f"{Path(image_path).stem}.json"
        with open(metadata_filename, "w") as file:
            json.dump(combined_dict, file, indent=4, default=str)
        
        log.info(f"Metadata saved to {metadata_filename}.\n\n\n")

    def _get_image_info(self, image_info: pd.DataFrame) -> dict:
        """
        This function extracts the image information from the dataframe.

        Parameters:
            image_info (pd.DataFrame): Dataframe containing the image information.

        Returns:
            dict: Extracted image information.
        """

        # Extract the relevant dataf from the image_info saved on the tablet when the image was taken
        image_info_list = [
            "Name", "Extension", "ImageURL", "UploadDateTimeUTC", "CameraInfo_DateTime", "SizeMiB", "HasMatchingJpgAndRaw", "ImageIndex", "UsState"
        ]

        image_info_imp = image_info[image_info_list].to_dict(orient='list')
        image_info_dict = {key: value[0] if value else None for key, value in image_info_imp.items()}
        image_info_dict["Name"] = image_info_dict["Name"].split(".")[0]

        image_info_dict = self._custom_decoder(image_info_dict) # deal with NaN values and en dash
        image_info_dict = self._replace_en_dash(image_info_dict) # deal with en dash

        return image_info_dict

    def _get_plant_field_info(self, image_info: pd.DataFrame) -> dict:
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

    def _get_category(self, image_name: str) -> dict:
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
                category.pop('alias') if 'alias' in category else None # delete class_id in the final metadata
                break
            
        return category

    def _get_exif_data(self, exif_data: dict) -> dict:
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

    def _get_bbox_xywh(self, detection_results: dict) -> dict:
        """
        This function extracts the bounding box coordinates.

        Parameters:
            detection_results (dict): Detection results.

        Returns:
            dict: Extracted bounding box coordinates.
        """
        if detection_results is not None:
            bbox_xywh = [detection_results["bbox"]["x_min"], detection_results["bbox"]["y_min"], detection_results["bbox"]["x_max"], detection_results["bbox"]["y_max"]]
            bbox_xywh_dict = {"bbox_xywh": bbox_xywh}
        else:
            bbox_xywh_dict = {"bbox_xywh": None} 

        return bbox_xywh_dict

    @staticmethod
    def _custom_decoder(data: dict) -> dict:
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
    def _replace_en_dash(data: dict) -> dict:
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
        self.output_dir = Path(cfg.data.temp_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.weed_detector = WeedDetector(cfg.data.path_yolo_model)
        self.metadata_extractor = MetadataExtractor(cfg)

        self.temp_dir = Path(cfg.data.temp_dir)

        self.process_batches()

    def process_batches(self) -> None:
        """
        This function processes the batches of images by detecting weeds and saving the metadata.
        """
        # Loop through the batches
        batches = list(Path(self.temp_dir).iterdir())
        for batch in batches:
            image_dir = Path(batch / "developed-images")
            self.image_loader = ImageLoader(image_dir)
            for image_path in image_dir.iterdir():
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPG", ".JPEG"}:
                    self.image_path = Path(image_path)
                    self.process_image(self.image_path)


    def process_image(self, image_path: Path) -> None:
        """
        This function processes a single image.
        """

        image = self.image_loader.read_image(self.image_path)
        
        # Create the metadata directory
        image_metadata_dir = image_path.parent.parent / 'cutouts'
        image_metadata_dir.mkdir(parents=True, exist_ok=True)

        
        class_id = self.metadata_extractor.get_class_id(image_path.name)
        
        if not class_id:
            log.warning(f"Class_id not found.")
            return
        
        # Detect weeds in the image
        detection_results = self.weed_detector.detect_weeds(image)
        if detection_results is not None:
            detection_results = self.weed_detector.detect_weeds(image)
            detection_results["image_id"] = Path(self.image_path).stem
            detection_results["class_id"] = class_id
        else: 
            log.warning(f"No detection.")
        
        exif_data = self.metadata_extractor.get_exif_data(str(self.image_path))

        # Save the image metadata
        self.metadata_extractor.save_image_metadata(str(self.image_path), detection_results, image_metadata_dir, exif_data)

def main(cfg: DictConfig) -> None:
    """
    Main function to start the weed detection process.

    Parameters:
        cfg (DictConfig): Configuration object containing the paths to the data files.

    Returns:
        None
    """
    log.info(f"Starting {cfg.general.task}")
    process_weeds = ProcessDetections(cfg)
    log.info(f"{cfg.general.task} completed.")
