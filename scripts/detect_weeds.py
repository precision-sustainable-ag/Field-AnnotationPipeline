import json
import logging
from PIL import Image
from PIL.ExifTags import TAGS
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from ultralytics import YOLO
from typing import Optional, Dict
import math

# Configure logging
log = logging.getLogger(__name__)


class ImageLoader:
    """
    A class for loading images from a given directory.
    """

    def __init__(self, image_dir: Path):
        self.image_dir = image_dir

    def read_image(self, image_path: str) -> Optional[np.ndarray]:
        """Reads an image from the specified path and converts it to an RGB numpy array."""
        try:
            log.info(f"Reading image from {image_path}.")
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            log.error(f"Failed to read image {image_path}: {e}", exc_info=True)
            return None


class WeedDetector:
    """
    A class for detecting weeds using a YOLO model.
    """

    def __init__(self, model_path: str):
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

            if not results:
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
        """Crops the detected region from the image based on the bounding box."""
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
        """Saves the image to the specified path."""
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
        Initializes the metadata extractor.

        Parameters:
            csv_path (str): Path to the CSV file containing metadata.
            field_species_info_path (str): Path to the field species info JSON file.
        """
        self.csv_path=cfg.data.merged_tables_permanent
        self.field_species_info_path=cfg.data.field_species_info

        self.metadata_dir = Path(cfg.data.image_metadata_dir)
        
        self.df = pd.read_csv(self.csv_path, low_memory=False)
        assert not self.df.empty, "Merged data tables CSV is empty."

        with open(self.field_species_info_path, "r") as file:
            self.field_species_info = json.load(file)

    def get_class_id(self, image_name: str) -> Optional[str]:
        """
        Gets the class ID using the image name.
        """
        log.info("Loading image data.")
        species_series = self.df[self.df["Name"] == image_name]["Species"]
        if species_series.empty:
            log.warning(f"Species data not found for image: {image_name}")
            return None

        species_list = [str(species).lower() for species in species_series]
        result = [item for item in self.field_species_info if item.get("common_name", "").lower() in species_list]
        
        return result[0]["class_id"] if result else None

    @staticmethod
    def get_exif_data(image_path: str) -> dict:
        """Extracts EXIF data from the image."""
        
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
        Generates and saves image metadata.

        Parameters:
            image_path (str): Path to the image file.
            detection_results (dict): Detection results including bbox.
            output_dir (Path): Directory to save the metadata JSON.
        """
        try:
            log.info("Extracting image metadata from tables.")
            image_name = Path(image_path).name
            image_info = self.df[self.df["Name"] == image_name]

            if image_info.empty:
                raise ValueError(f"Image '{image_name}' not found in the dataframe.")

            specific_data_list = [
                "Name", "UploadDateTimeUTC", "MasterRefID", "ImageURL", "CameraInfo_DateTime", "SizeMiB", "ImageIndex", "UsState", "PlantType", "CloudCover",
                "GroundResidue", "GroundCover", "Username", "CoverCropFamily", "GrowthStage", "CottonVariety", "CropOrFallow",
                "CropTypeSecondary", "Species", "Height", "SizeClass", "FlowerFruitOrSeeds", "BaseName", "Extension", "HasMatchingJpgAndRaw"
            ]

        
            image_info_imp = image_info[specific_data_list].to_dict(orient='list')
            image_info_imp_dict = {key: value[0] if value else None for key, value in image_info_imp.items()}

            image_info_imp_dict = self._custom_decoder(image_info_imp_dict)
            image_info_imp_dict = self._replace_en_dash(image_info_imp_dict)

            combined_dict = {
                "detection_results": detection_results,
                "image_info": image_info_imp_dict,
                "exif_info": exif_data
            }

            metadata_filename = self.metadata_dir / f"{Path(image_path).stem}.json"
            with open(metadata_filename, "w") as file:
                json.dump(combined_dict, file, indent=4, default=str)
            
            log.info(f"Metadata saved to {metadata_filename}.")
        
        except Exception as e:
            log.error(f"Failed to extract image metadata for {image_path}: {e}", exc_info=True)

    @staticmethod
    def _custom_decoder(data: dict) -> dict:
        """
        Custom decoder to handle NaN values.

        Parameters:
            data (dict): Dictionary to decode.

        Returns:
            dict: Decoded dictionary.
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
        Replace en dash with hyphen in string values.

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
        Initializes directories and components for the weed detection process.

        Parameters:
            cfg (DictConfig): Configuration object containing directory paths and model details.
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
        Processes a single image: detects weeds, crops the image, and extracts metadata.

        Parameters:
            image_path (Path): Path to the image file.
        """
        try:
            image = self.image_loader.read_image(image_path)
            if image is None:
                return

            class_id = self.metadata_extractor.get_class_id(image_path.name)
            if not class_id:
                return

            detection_results = self.weed_detector.detect_weeds(image)
            detection_results["image_id"] = Path(image_path).stem
            detection_results["class_id"] = class_id
            if detection_results is None:
                return

            cropout_image = self.image_processor.crop_image(image, detection_results['bbox'])
            if cropout_image is not None:
                cropout_image_path = self.output_dir / f"{image_path.stem}_cropout.jpg"
                self.image_processor.save_image(cropout_image, cropout_image_path)

            exif_data = self.metadata_extractor.get_exif_data(str(image_path))
            self.metadata_extractor.save_image_metadata(str(image_path), detection_results, self.image_metadata_dir, exif_data)

        except Exception as e:
            log.error(f"Error processing image {image_path}: {e}", exc_info=True)

    def process_all_images(self) -> None:
        """
        Processes all images in the directory specified in the configuration.
        """
        for image_path in self.image_loader.image_dir.iterdir():
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPG", ".JPEG"}:
                self.process_image(image_path)


def main(cfg: DictConfig) -> None:
    """
    Main function to start the weed detection task.

    Parameters:
        cfg (DictConfig): Configuration object containing the task details.
    """
    log.info(f"Starting {cfg.general.task}")
    process_weeds = ProcessDetections(cfg)
    process_weeds.process_all_images()
    log.info(f"{cfg.general.task} completed.")
