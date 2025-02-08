import json
import logging
import yaml
import cv2
import numpy as np
import pandas as pd
import math
import exifread
import copy 
import os

from PIL import Image
from PIL.ExifTags import TAGS
from pathlib import Path
from omegaconf import DictConfig
from ultralytics import YOLO
from typing import Optional, Dict
from tqdm import tqdm

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

        Parameters:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Image as a numpy array.
        """
        log.info(f"Reading image from {image_path}.")
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

        Returns:
            None
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
        log.info("Starting weed detection.")
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

class ImageProcessor:
    """
    A class for processing images, including cropping and saving.
    """

    @staticmethod
    def crop_image(image: np.ndarray, bbox: Dict[str, int]) -> Optional[np.ndarray]:
        """
        This function crops the detected region from the image based on the bounding box.

        Parameters:
            image (np.ndarray): Image as a numpy array.
            bbox (dict): Bounding box coordinates.

        Returns:  
            np.ndarray: Cropped image.
        """
        x_min, y_min, x_max, y_max = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
        cropout_image = image[y_min:y_max, x_min:x_max]
        log.info("Image cropping completed.")
        return cropout_image

    @staticmethod
    def save_image(image: np.ndarray, image_path: Path) -> None:
        """
        Saves the image to the specified path.
        
        Parameters:
            image (np.ndarray): Image as a numpy array.
            image_path (Path): Path to save the image.

        Returns:
            None
        """
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        log.info(f"Image saved to {image_path}.")


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
        self.csv_path = cfg.paths.merged_tables_permanent
        self.species_info_path = cfg.paths.field_species_info
        self.metadata_version = cfg.metadata_version
        self.df = pd.read_csv(self.csv_path, low_memory=False)
        assert not self.df.empty, "Merged data tables CSV is empty."

        with open(self.species_info_path, "r") as file:
            self.species_info = json.load(file)

    def get_class_id(self, image_name: str) -> Optional[str]:
        """
        This function extracts the class ID from the image name.

        Parameters:
            image_name (str): Name of the image.

        Returns:
            str: Class ID of the species if found; otherwise, returns None.
        """
        log.info(f"Loading image data for {image_name}.")

        # Find the species for the given image
        species_series = self.df[self.df["Name"].str.lower() == image_name.lower()]["Species"]
        if species_series.empty:
            log.warning(f"Species data not found for image: {image_name}")
            return None
        
        species = [str(species).lower() for species in species_series][0]
        class_id = self._find_class_id(species)
        log.info(f"Class ID: {class_id}")

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

        The function reads the image file, processes the EXIF tags using exifread, 
        formats the metadata into a dictionary, and converts it to an ImageMetadata instance.

        Returns:
            ImageMetadata: The extracted EXIF metadata.
        """
        log.info("Extracting EXIF data from the image.")

        # Open image file for reading (must be in binary mode)
        f = open(image_path, "rb")
        
        # Return Exif tags
        tags = exifread.process_file(f, details=False)
        f.close()

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

        # # Create an instance of ImageMetadata with the processed metadata
        # imgmeta = ImageMetadata(**meta)
        return filtered_exif_data
        
    def save_image_metadata(self, image_path: str, detection_results: dict, image_metadata_dir: Path, exif_data: dict) -> None:
        """
        This function saves the metadata extracted from the image.
        
        Parameters:
            image_path (str): Path to the image file.
            detection_results (dict): Detection results.
            output_dir (Path): Directory to save the metadata.
            exif_data (dict): Extracted EXIF data.

        Returns:
            None
        """
        log.info("Extracting metadata")

        # Extract the image information, plant field information, category, and relevant EXIF data
        image_name = Path(image_path).name
        image_info = self.df[self.df["Name"].str.lower() == image_name.lower()]
        if image_info.empty:
            raise ValueError(f"Image '{image_name}' not found in the dataframe.")

        image_info_dict = self._get_image_info(image_info)
        plant_field_info_dict = self._get_plant_field_info(image_info)
        category = self._get_category(image_name)

        for key in ('collection_location', 'collection_timing'):
            category.pop(key, None)  # `None` prevents KeyError if key doesn't exist

        exif_data_imp_dict = self._get_exif_data(exif_data)

        # Combine the extracted metadata into a single dictionary
        combined_dict = {
            "image_info": image_info_dict,
            "plant_field_info": plant_field_info_dict,
            "annotation": self._get_bbox_xywh(detection_results),
            "category": category,
            "exif_meta": exif_data_imp_dict,
            "version": self.metadata_version
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
        # Extract the bounding box coordinates
        if detection_results is not None:
            bbox_height = detection_results["bbox"]["y_max"] - detection_results["bbox"]["y_min"]
            bbox_width = detection_results["bbox"]["x_max"] - detection_results["bbox"]["x_min"]
            bbox_xywh = [detection_results["bbox"]["x_min"], detection_results["bbox"]["y_min"], bbox_width, bbox_height]
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
        self.output_dir = Path(cfg.paths.temp_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.weed_detector = WeedDetector(cfg.paths.yolo_weed_detection_model)
        self.image_processor = ImageProcessor()
        self.metadata_extractor = MetadataExtractor(cfg)

        # Loop through the batches
        batches = list(Path(cfg.paths.temp_dir).iterdir())
        for batch in batches:
            image_dir = Path(batch /"developed-images")
            self.image_loader = ImageLoader(image_dir)

            # Loop through the images in the batch
            for image_path in image_dir.iterdir():
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPG", ".JPEG"}:
                    self.image_path = Path(image_path)
                    self.process_image(self.image_path)

    def process_image(self, image_path: Path) -> None:
        """
        This function processes the image by detecting weeds and saving the metadata.

        Parameters:
            image_path (Path): Path to the image file.

        Returns:
            None
        """
        image = self.image_loader.read_image(self.image_path)

        image_metadata_dir = Path(os.path.join(os.path.dirname(os.path.dirname(image_path)), 'cutouts')) # save metadata in the same batch as the image

        # Create the metadata directory if it doesn't exist
        os.makedirs(image_metadata_dir, exist_ok=True)

        if image is None:
            log.warning(f"No image present for processing.")
            return

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
    ProcessDetections(cfg)
    log.info(f"{cfg.general.task} completed.")
