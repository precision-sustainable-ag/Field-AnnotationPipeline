import json
import logging
import yaml
import cv2
import numpy as np
import pandas as pd
import math

from PIL import Image
from PIL.ExifTags import TAGS
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

        Parameters:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Image as a numpy array.
        """
        try:
            log.info(f"Reading image from {image_path}.")
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            log.error(f"Failed to read image {image_path}: {e}", exc_info=True)
            return None

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
        try:
            results = self.model(image)
            
            if not results or not results[0].boxes.xyxy.tolist():
                log.warning("No detection found.")
                return None

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

        except Exception as e:
            log.error(f"Failed to detect weeds: {e}", exc_info=True)
            return None


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
        try:
            x_min, y_min, x_max, y_max = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
            cropout_image = image[y_min:y_max, x_min:x_max]
            log.info("Image cropping completed.")
            return cropout_image
        except Exception as e:
            log.error(f"Failed to crop image: {e}", exc_info=True)
            return None

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
        try:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            log.info(f"Image saved to {image_path}.")
        except Exception as e:
            log.error(f"Failed to save image to {image_path}: {e}", exc_info=True)


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
        self.csv_path = cfg.data.merged_tables_permanent
        self.species_info_path = cfg.data.species_info

        with open(cfg.morphology_species, 'r') as f:
            self.broad_sprase_morph_species = yaml.safe_load(f)

        self.metadata_dir = Path(cfg.data.image_metadata_dir)
        
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
        log.info("Loading image data.")
        species_series = self.df[self.df["Name"] == image_name]["Species"]
        if species_series.empty:
            log.warning(f"Species data not found for image: {image_name}")
            return None
        
        species = [str(species).lower() for species in species_series]
        class_id = [value['class_id'] for _, value in self.species_info['species'].items() if value['common_name'].lower() in species][0]
        return class_id

    @staticmethod
    def get_exif_data(image_path: str) -> dict:
        """
        This function extracts EXIF data from the image.
        
        Parameters:
            image_path (str): Path to the image file.
        
        Returns:
            dict: Extracted EXIF data.
        """
        
        try:
            log.info("Extracting EXIF data from the image.")
            image = Image.open(image_path)
            exclude_keys = ["PrintImageMatching", "UserComment", "MakerNote", "ComponentsConfiguration", "SceneType"]

            filtered_exif_data = {}
            if hasattr(image, '_getexif'):
                exif_info = image._getexif()
                if exif_info:
                    filtered_exif_data = {TAGS.get(tag, tag): value for tag, value in exif_info.items() if TAGS.get(tag, tag) not in exclude_keys}
            log.info("Extracting EXIF data from the image completed.")
            return filtered_exif_data

        except Exception as e:
            log.error(f"Failed to extract EXIF data from {image_path}: {e}", exc_info=True)
            return {}

    def save_image_metadata(self, image_path: str, detection_results: dict, output_dir: Path, exif_data: dict) -> None:
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
        try:
            log.info("Extracting metadata")
            # save detailed species info in "category" of metadata
            category = {}
            if detection_results is not None:
                for species_key, species_value in self.species_info['species'].items():
                        if detection_results['class_id'] == species_value['class_id']:
                            category = species_value
                            break
            else:
                log.warning("Nothing to add to category.")
                category = None

            image_name = Path(image_path).name
            image_info = self.df[self.df["Name"] == image_name]

            if image_info.empty:
                raise ValueError(f"Image '{image_name}' not found in the dataframe.")

            # add image_info data to metadata
            image_info_list = [
                "Name", "Extension", "ImageURL", "UploadDateTimeUTC", "CameraInfo_DateTime", "SizeMiB", "HasMatchingJpgAndRaw", "ImageIndex", "UsState"
            ]

            image_info_imp = image_info[image_info_list].to_dict(orient='list')
            image_info_dict = {key: value[0] if value else None for key, value in image_info_imp.items()}
            image_info_dict["Name"] = image_info_dict["Name"].split(".")[0]

            image_info_dict = self._custom_decoder(image_info_dict) # deal with NaN values and en dash
            image_info_dict = self._replace_en_dash(image_info_dict) # deal with en dash

            # add plant_field_info data to metadata
            plant_field_info_list = [
                "PlantType", "CloudCover", "GroundResidue", "GroundCover", "CoverCropFamily", "GrowthStage", "CottonVariety", "CropOrFallow",
                "CropTypeSecondary", "Species", "Height", "SizeClass", "FlowerFruitOrSeeds"
            ]

            plant_field_info = image_info[plant_field_info_list].to_dict(orient='list')
            plant_field_info_dict = {key: value[0] if value else None for key, value in plant_field_info.items()}

            plant_field_info_dict = self._custom_decoder(plant_field_info_dict) # deal with NaN values and en dash
            plant_field_info_dict = self._replace_en_dash(plant_field_info_dict) # deal with en dash

            # add selected exif data to metadata
            exif_data_list = [
                "ImageWidth", "ImageLength", "Make", "Model", "Software", "DateTime", "ExposureTime", "FNumber", "ExposureProgram", "ISOSpeedRatings", 
                "RecommendedExposureIndex", "ExifVersion", "BrightnessValue", "MaxApertureValue", "LightSource", "Flash", "FocalLength", "ExposureMode", 
                "WhiteBalance", "FocalLengthIn35mmFilm", "Contrast", "Saturation", "Sharpness", "LensModel", "LensSpecification", "BodySerialNumber"
            ]

            exif_data_imp_dict = {key: exif_data[key] for key in exif_data_list if key in exif_data}

            exif_data_imp_dict = self._custom_decoder(exif_data_imp_dict) # deal with NaN values and en dash
            exif_data_imp_dict = self._replace_en_dash(exif_data_imp_dict) # deal with en dash

            # extract bbox_xywh from detection_results
            if detection_results is not None:
                bbox_xywh = [detection_results["bbox"]["x_min"], detection_results["bbox"]["y_min"], detection_results["bbox"]["x_max"], detection_results["bbox"]["y_max"]]
                bbox_xywh_dict = {"bbox_xywh": bbox_xywh}
            else:
                bbox_xywh_dict = None

            # combine all metadata into a single dictionary
            combined_dict = {
                "image_info": image_info_dict,
                "plant_field_info": plant_field_info_dict,
                "annotation": bbox_xywh_dict,
                "category": category,
                "exif_info": exif_data_imp_dict
            }

            metadata_filename = self.metadata_dir / f"{Path(image_path).stem}.json"
            self.metadata_dir.mkdir(exist_ok=True, parents=True)

            with open(metadata_filename, "w") as file:
                json.dump(combined_dict, file, indent=4, default=str)
            
            log.info(f"Metadata saved to {metadata_filename}.")
            
        except Exception as e:
            log.error(f"Failed to extract metadata for {image_path}: {e}", exc_info=True)

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
        
        self.image_metadata_dir = Path(cfg.data.image_metadata_dir)
        self.image_loader = ImageLoader(Path(cfg.data.temp_image_dir))
        self.weed_detector = WeedDetector(cfg.data.path_yolo_model)
        self.image_processor = ImageProcessor()
        self.metadata_extractor = MetadataExtractor(cfg)

    def process_image(self, image_path: Path) -> None:
        """
        THis function processes the image by detecting weeds and saving the metadata.

        Parameters:
            image_path (Path): Path to the image file.

        Returns:
            None
        """
        try:
            image = self.image_loader.read_image(image_path)
            if image is None:
                return

            class_id = self.metadata_extractor.get_class_id(image_path.name)
            if not class_id:
                return
            
            detection_results = self.weed_detector.detect_weeds(image)
            if detection_results is not None:
                detection_results = self.weed_detector.detect_weeds(image)
                detection_results["image_id"] = Path(image_path).stem
                detection_results["class_id"] = class_id
            else: 
                log.warning(f"No detection.")
            
            exif_data = self.metadata_extractor.get_exif_data(str(image_path))
            self.metadata_extractor.save_image_metadata(str(image_path), detection_results, self.image_metadata_dir, exif_data)

        except Exception as e:
            log.error(f"Error processing image {image_path}: {e}", exc_info=True)

    def process_all_images(self) -> None:
        """
        This function processes all images in the specified directory.

        Returns:
            None
        """
        for image_path in self.image_loader.image_dir.iterdir():
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPG", ".JPEG"}:
                self.process_image(image_path)


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
    process_weeds.process_all_images()
    log.info(f"{cfg.general.task} completed.")
