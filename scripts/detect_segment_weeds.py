import json
import logging
from skimage import segmentation, measure
import os
from PIL import Image
from PIL.ExifTags import TAGS
from pathlib import Path
from utils.utils import add_padding_to_cropout, shifting_bbox_coordinates

import cv2
import math
import numpy as np
import pandas as pd
from omegaconf import DictConfig
# from segment_anything import SamPredictor, sam_model_registry
from segment_anything_hq import SamPredictor, sam_model_registry
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class SegmentWeeds:
    """
     A class to handle weed detection and segmentation of images from the Field-Image Repository.

    This class outputs:
    - Image cropout
    - Image cutout
    - Cutout mask
    - Cutout metadata: image_info, cutout_props, category, and exif_meta 
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes directories and model paths.

        Parameters:
            cfg (DictConfig): Configuration object containing directory paths and model details.
        """
        self.image_dir = Path(cfg.data.temp_image_dir)
        self.output_dir = Path(cfg.data.temp_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_metadata_dir = Path(cfg.data.image_metadata_dir)
        self.specific_data_json = cfg.data.specific_data

        self.path_yolo_model = cfg.data.path_yolo_model
        self.sam_checkpoint = cfg.data.sam_checkpoint
        self.sam_hq_checkpoint = cfg.data.sam_hq_checkpoint
        self.sam_model_type = cfg.data.sam_model_type

        self.csv_path = cfg.data.merged_tables
        self.df = pd.read_csv(self.csv_path, low_memory=False)

        with open(cfg.data.field_species_info, "r") as file:
            self.field_species_info = json.load(file)

        with open(cfg.data.species_info, "r") as file:
            self.species_info = json.load(file)

    def load_image_data(self, image_path: str) -> str:
        """
        Loads data for the given image.

        Parameters:
            image_path (str): Path to the image file.

        Returns:
            str: The class ID if species is found, None otherwise.
        """
        try:
            log.info("Loading image data.")
            image_name = os.path.basename(image_path)
            species_series = self.df[self.df["Name"] == image_name]["Species"]

            if not species_series.empty:
                species_list = [str(species).lower() for species in species_series]
                for item in self.field_species_info:
                    if str(item["common_name"]).lower() in species_list:
                        return item["class_id"]
            else:
                log.warning(f"Species data not found for image: {image_name}")
                return None
        except Exception as e:
            log.error(f"Error loading image data for {image_path}: {e}", exc_info=True)
            return None
        log.info("Loading image data completed.")

    def detect_target_weed(self, image_path: str):
        """
        Detects target weed in the given image.

        Parameters:
            image_path (str): Path to the image file.

        Returns:
            tuple: cropout_image, bbox, and original image
        """
        log.info("Starting weed detection.")
        try:
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

            # cropped_image = image_cropping(image, (1024, 1024))  # image cropped to 1024 x 1024
            # print(f"Shape of cropped image: {cropped_image.shape}")

            yolo_model = YOLO(self.path_yolo_model)
            results = yolo_model(image)

            if not results:
                log.warning(f"No detection for image: {image_path}")
                return

            bbox = results[0].boxes.xyxy.tolist()[0]
            x_min, y_min, x_max, y_max = map(int, bbox)

            # print(f"Shape of bounding box: x_min:{x_min}, x_max:{x_max}, y_min:{y_min}, y_max:{y_max}")
            # print(f"Shape of bounding box: {x_max-x_min}, {y_max-y_min}")

            cropout_image = image[y_min:y_max, x_min:x_max]
            cropout_image_rgb = cv2.cvtColor(cropout_image, cv2.COLOR_BGR2RGB)

            cropout_image_path = self.output_dir / f"{image_path.stem}_cropout.jpg"
            cv2.imwrite(str(cropout_image_path), cropout_image_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            # cropped_image_path = self.output_dir / f"{image_path.stem}_cropped_image.png"
            # cv2.imwrite(str(cropped_image_path), cropped_image)

            return cropout_image, bbox, image

        except Exception as e:
            log.error(f"Failed to process {image_path}: {e}", exc_info=True)
        log.info(f"Completed weed detection.")

    def segment_target_weed(self, image_path, bbox, class_id, cropout_image, image):
        """
        Segment target weed in the given image.

        Parameters:
            image_path (str): Path to the image file.
            class_id (str): Class ID for the detected species.

        Returns:
            tuple: Cropped mask and cropped image.
        """
        log.info(f"Starting weed segmentation for {image_path.stem}.")
        try:            
            padded_image, padding = add_padding_to_cropout(cropout_image)

            top_padding = padding['top']
            left_padding = padding['left']

            # original bbox location
            x_min, y_min, x_max, y_max = map(int, bbox)
            input_box_og = np.array([x_min, y_min, x_max, y_max])

            # calculate bbox location in the padded image
            new_x_min = int(left_padding)
            new_y_min = int(top_padding)
            new_x_max = int(left_padding + (x_max - x_min))
            new_y_max = int(top_padding + (y_max - y_min) )

            input_box = np.array([new_x_min, new_y_min, new_x_max, new_y_max])

            # # Check if coordinates are correct
            if cropout_image.all() == padded_image[new_y_min:new_y_max, new_x_min:new_x_max].all():
                log.info("The modified Bbox location is correct")
            else:
                log.warning("The modified Bbox location is not correct")
        
            # # Replace black color (0, 0, 0) with NaN
            # # Create a mask where all black pixels are True
            # black_pixels_mask = np.all(padded_image == [0, 0, 0], axis=-1)

            # # Convert image to float to allow NaN values
            # padded_image = padded_image.astype(float)

            # # Replace black pixels with NaN
            # padded_image[black_pixels_mask] = np.nan

            device = "cuda"
            sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_hq_checkpoint)
            sam.to(device=device)
            predictor = SamPredictor(sam)
            predictor.set_image(padded_image)

            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

            if masks is None or len(masks) == 0:
                log.error("No masks were generated by the SAM predictor.")
                return None

            cutout_mask = (masks[0] > 0.5).astype(np.uint8) 
            #cutout_mask_cropped = cutout_mask[y_min: y_max, x_min: x_max]
            cutout_mask_cropped = cutout_mask[new_y_min: new_y_max, new_x_min: new_x_max]
            new_mask = cutout_mask_cropped.copy()
            new_mask[new_mask == 1] = class_id
            # new_mask[new_mask == 1] = 255

            cutout_image = cv2.bitwise_and(cropout_image, cropout_image, mask=new_mask)
            cutout_image = cv2.cvtColor(cutout_image, cv2.COLOR_BGR2RGB)
            cutout_image_path = self.output_dir / f"{image_path.stem}_cutout.png"
            cv2.imwrite(str(cutout_image_path), cutout_image)

            cutout_mask_path = self.output_dir / f"{image_path.stem}_cutout_mask.png"
            cv2.imwrite(str(cutout_mask_path), new_mask)

            return cutout_mask, cutout_image

        except Exception as e:
            log.error(f"Error in weed segmentation: {e}", exc_info=True)
            return None
        finally:
            log.info("Completed target weed segmentation.")

    def get_exif_data(self, image_path: str) -> dict:
        """
        Extracts EXIF data from the image.

        Parameters:
            image_path (str): Path to the image file.

        Returns:
            dict: Filtered EXIF data.
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

    def get_image_metadata(self, filtered_exif_data: dict, image_path: str, class_id: str, cutout_props: dict) -> None:
        """
        Generates and saves image metadata.

        Parameters:
            filtered_exif_data (dict): EXIF data of the image.
            image_path (str): Path to the image file.
            class_id (str): Class ID for the detected species.
            cutout_props (dict): Properties of the cutout.
        """
        try:
            log.info("Extracting image metadata from tables.")
            image_name = os.path.basename(image_path)
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
            image_info_imp_dict["cutout_id"] = f"{Path(image_name).stem}_cutout.png"

            image_info_imp_dict = self._custom_decoder(image_info_imp_dict)
            image_info_imp_dict = self._replace_en_dash(image_info_imp_dict)
            cutout_props = self._custom_decoder(cutout_props)
            cutout_props = self._replace_en_dash(cutout_props)

            species_info = {value["class_id"]: value for value in self.species_info["species"].values()}
            image_species_info = species_info.get(class_id, {})

            combined_dict = {
                "image_info": image_info_imp_dict,
                "cutout_props": cutout_props,
                "category": image_species_info,
                "exif_meta": filtered_exif_data
            }

            metadata_filename = self.image_metadata_dir / f"{Path(image_path).stem}_metadata.json"
            with open(metadata_filename, "w") as file:
                json.dump(combined_dict, file, indent=4, default=str)
            log.info("Extracting image metadata from tables completed.")
        except Exception as e:
            log.error(f"Failed to extract image metadata for {image_path}: {e}", exc_info=True)

    def cutout_props(self, cutout_image: np.ndarray, cutout_mask_cropped: np.ndarray) -> dict:
        """
        Calculate properties of the cutout.

        Parameters:
            cutout_image (np.ndarray): The cropped out image.
            cutout_mask_cropped (np.ndarray): The cropped out mask.

        Returns:
            dict: Properties of the cutout.
        """
        log.info("Calculating image properties.")
        try:
            image_gray = cv2.cvtColor(cutout_image, cv2.COLOR_BGR2GRAY)
            image_hsv = cv2.cvtColor(cutout_image, cv2.COLOR_BGR2HSV)

            # calculate green sum
            lower_green_hsv = np.array([40, 70, 120])
            upper_green_hsv = np.array([90, 255, 255])
            green_hsv_mask = cv2.inRange(image_hsv, lower_green_hsv, upper_green_hsv)
            green_hsv_mask = np.where(green_hsv_mask == 255, 1, 0)
            green_sum = int(np.sum(green_hsv_mask))

            # calculate number of components
            if len(cutout_mask_cropped.shape) > 2:
                mask = cutout_mask_cropped[..., 0]
            else:
                mask = cutout_mask_cropped
            _, num_components = measure.label(mask, background=0, connectivity=2, return_num=True)

            # calculate other cutout properties
            _, image_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
            image_contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image_contours_largest = max(image_contours, key=cv2.contourArea)
            image_ellipse = cv2.fitEllipse(image_contours_largest)
            major_axis_length = max(image_ellipse[1])
            minor_axis_length = min(image_ellipse[1])
            eccentricity = np.sqrt(1 - (minor_axis_length**2 / major_axis_length**2))
            image_area = cv2.contourArea(image_contours_largest)
            hull = cv2.convexHull(image_contours_largest)
            hull_area = cv2.contourArea(hull)
            solidity = image_area / hull_area if hull_area > 0 else 0
            perimeter = cv2.arcLength(image_contours_largest, True)
            blur_effect = measure.blur_effect(image_gray)
            (B, G, R) = cv2.split(cutout_image)

            mean_B = np.mean(B)
            mean_G = np.mean(G)
            mean_R = np.mean(R)
            std_B = np.std(B)
            std_G = np.std(G)
            std_R = np.std(R)
            cropout_rgb_mean = [mean_R, mean_G, mean_B]
            cropout_rgb_std = [std_R, std_G, std_B]
            clear_border = segmentation.clear_border(cutout_mask_cropped)
            extends_border = not np.array_equal(cutout_mask_cropped, clear_border)

            cutout_props = {
                "area": image_area,
                "eccentricity": eccentricity,
                "solidity": solidity,
                "perimeter": perimeter,
                "green_sum": green_sum,
                "blur_effect": blur_effect,
                "num_components": num_components,
                "cropout_rgb_mean": cropout_rgb_mean,
                "cropout_rgb_std": cropout_rgb_std,
                "extends_border": extends_border
            }

            log.info("Calculating image properties completed.")
            return cutout_props

        except Exception as e:
            log.error(f"Failed to calculate cutout properties: {e}", exc_info=True)
            return {}

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
                data[key] = SegmentWeeds._custom_decoder(value)
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
                data[key] = SegmentWeeds._replace_en_dash(value)
        return data

def main(cfg: DictConfig) -> None:
    """
    Main function to start the weed detection and segmentation task.

    Parameters:
        cfg (DictConfig): Configuration object containing the task details.
    """
    log.info(f"Starting {cfg.general.task}")
    segment_weeds = SegmentWeeds(cfg)

    for image_path in segment_weeds.image_dir.iterdir():
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            try:
                class_id = segment_weeds.load_image_data(image_path)
                if class_id:
                    cropout_image, bbox, image = segment_weeds.detect_target_weed(image_path)
                    segment_weeds.segment_target_weed(image_path, bbox, class_id, cropout_image, image)
                    # cutout_mask_cropped, cutout_image = segment_weeds.segment_target_weed(image_path, bbox, class_id, cropout_image)
                    # exif_data = segment_weeds.get_exif_data(image_path)
                    # cutout_props = segment_weeds.cutout_props(cutout_image, cutout_mask_cropped)
                    # segment_weeds.get_image_metadata(exif_data, image_path, class_id, cutout_props)
            except Exception as e:
                log.error(f"Error processing image {image_path}: {e}", exc_info=True)

    log.info(f"{cfg.general.task} completed.")
