import os
import json
import math
import logging
import random
import cv2
import yaml
import numpy as np
import torch
import torch.multiprocessing as mp

from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
from omegaconf import DictConfig
from skimage.measure import regionprops
from skimage.morphology import remove_small_holes, label, remove_small_objects
from skimage.morphology import disk, dilation, erosion
from skimage.filters import gaussian
from segment_anything_hq import sam_model_registry, SamPredictor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

device = "cuda"

class SingleImageProcessor:
    def __init__(self, cfg: DictConfig, output_dir: str, metadata_dir: str, model_type: str, sam_checkpoint: str) -> None:
        """
        Initialize the SingleImageProcessor with the configuration and output directories.

        Parameters:
            cfg (DictConfig): The configuration object.
            output_dir (str): The directory where the output files will be saved.
            metadata_dir (str): The directory where the metadata files are stored.
            model_type (str): The type of model to use for segmentation.
            sam_checkpoint (str): The path to the SAM model checkpoint.

        Returns:
            None
        """
        log.info("Initializing SingleImageProcessor")
        self.metadata_dir = Path(metadata_dir)
        self.output_dir = Path(output_dir)
        self.visualization_label_dir = self.output_dir / "vis_masks" 

        self.broad_spceies = cfg.morphology.broad_morphology
        self.sparse_spceies = cfg.morphology.sparse_morphology

        for output_dir in [self.output_dir, self.visualization_label_dir]:
            output_dir.mkdir(exist_ok=True, parents=True)

        log.info("Loading SAM model")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.mask_predictor = SamPredictor(sam)

    def process_image(self, input_paths: Tuple[str, str]) -> Tuple[dict, dict]:
        """
        Processes an image and its JSON annotations to extract bounding box information.
        Parameters:
            input_paths (Tuple[str, str]): 
                - Path to the image file.
                - Path to the JSON file with annotations.

        Returns:
            Tuple[dict, dict]: 
                - JSON data dictionary.
                - Bounding box information with `image_id`, `class_id`, `x_min`, `y_min`, `x_max`, `y_max`, `width`, and `height`.
        """
        image_path, json_path = input_paths
        log.info(f"Processing image: {image_path}")
        
        if Path(json_path).exists():
            with open(json_path, 'r') as f:
                data = json.load(f)

            if data["annotation"]["bbox_xywh"] is not None:
                image_id = data["image_info"]["Name"]
                class_id = data["category"]["class_id"]
                bbox = data["annotation"]["bbox_xywh"]

                # Internal bbox structure to include image_id, class_id, and different format for bbox
                _bbox = {
                    "image_id": image_id,
                    "class_id": class_id,
                    "x_min": bbox[0], "y_min": bbox[1],
                    "x_max": bbox[2], "y_max": bbox[3],
                    "width": bbox[2] - bbox[0],
                    "height": bbox[3] - bbox[1]
                }

                self._find_bbox_center(_bbox)
                return data, _bbox
            else:
                log.error(f"No detection results found in JSON file: {json_path}")
                return None
            
        else:
            log.error(f"No JSON file found for {json_path}")
            return None

    def save_cutout(self, input_paths: Tuple[str, str]) -> None:
        """
        Saves the final cutout, cropped image, and mask.

        Parameters:
            input_paths (Tuple[str, str]): Paths to the input image and JSON file.
        
        Returns:
            None
        """
        log.info("Starting process to save cropout, final mask, and cutout.")
        
        if self.process_image(input_paths) is not None:
            data, _bbox = self.process_image(input_paths)
            # Process image to get bounding box and data
            image_path, _ = input_paths
            # Read the image
            image = self._read_image(image_path)
            # Create and process masks
            masked_image, class_masked_image = self._create_masks(image, _bbox)
            class_masked_image = class_masked_image.astype(np.uint8)
            class_masked_image_cropped = self._crop_image(class_masked_image, _bbox['y_min'], _bbox['y_max'], _bbox['x_min'], _bbox['x_max'])

            # Convert the mask to 3D
            class_masked_image_3d = np.repeat(class_masked_image[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
            
            # Directory setup for saving outputs
            class_id = data["category"]["class_id"]
            save_class_dir = Path(self.output_dir, f"{class_id}")
            save_class_dir.mkdir(exist_ok=True, parents=True)
            log.info(f"Output directory created at: {save_class_dir}")
            
            # Create the final cutout image
            final_cutout_bgr = np.where(class_masked_image_3d != 255, image, 0)
            final_cutout_rgb = cv2.cvtColor(final_cutout_bgr, cv2.COLOR_BGR2RGB)
            final_cutout_rgb = self._crop_image(final_cutout_rgb, _bbox['y_min'], _bbox['y_max'], _bbox['x_min'], _bbox['x_max'])

            # Save cropped image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_cropped = self._crop_image(image_rgb, _bbox['y_min'], _bbox['y_max'], _bbox['x_min'], _bbox['x_max'])
            cropout_name = Path(image_path).stem + '_cropout.png'
            cv2.imwrite(str(save_class_dir / cropout_name), image_cropped.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            log.info(f"Cropped image saved as: {cropout_name}")
            
            # Save the final mask
            class_masked_image_cropped = np.where(class_masked_image_cropped == 255, 0, class_masked_image_cropped)
            final_mask_name = Path(image_path).stem + '_mask.png'
            self.save_compressed_image(masked_image, self.visualization_label_dir / final_mask_name)
            cv2.imwrite(str(save_class_dir / final_mask_name), class_masked_image_cropped.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            log.info(f"Final mask saved as: {final_mask_name}")

            # Save the final cutout
            cutout_name = Path(image_path).stem + '_cutout.png'
            cv2.imwrite(str(save_class_dir / cutout_name), final_cutout_rgb.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            log.info(f"Final cutout saved as: {cutout_name}")
        else:
            log.error(f"Data or _bbox is None, skipping image processing.")

    def save_compressed_image(self, image: np.ndarray, path: str, quality: int = 98) -> None:
        """
        Save the image in a compressed JPEG format.

        Parameters:
            image (np.ndarray): The image to save, expected in RGB format.
            path (str): The file path where the image will be saved.
            quality (int, optional): The quality of the JPEG compression (0 to 100). Default is 98.
        
        Returns:
            None
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        is_success, encoded_image = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if is_success:
            with open(path, 'wb') as f:
                encoded_image.tofile(f)

    def _find_bbox_center(self, _bbox: dict) -> None:
        """
        Calculate and add the center coordinates of the bounding box to the dictionary.

        Parameters:
            bbox (dict): Dictionary containing bounding box coordinates with keys 
                        'x_min', 'x_max', 'y_min', and 'y_max'.
                    
        Returns:
            None
        """
        _bbox['center_x'] = (_bbox['x_min'] + _bbox['x_max']) / 2
        _bbox['center_y'] = (_bbox['y_min'] + _bbox['y_max']) / 2

    def _read_image(self, image_path: Path) -> np.ndarray:
        """
        Read an image from a specified path and convert it to RGB format.

        Parameters:
            image_path (Path): Path to the image file.

        Returns:
            np.ndarray: The image in RGB format.
        """
        log.info(f"Reading image and converting to RGB: {image_path}")
        return cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

    def _create_masks(self, image: np.ndarray, _bbox: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create masked images based on the given bounding box annotations.

        Parameters:
            image (np.ndarray): The input image in RGB format.
            _bbox (dict): Dictionary containing bounding box coordinates and other annotations.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing two arrays:
                                            - Masked image with RGBA channels.
                                            - Class mask image (binary mask).
        """
        im_size_X, im_size_Y = image.shape[1], image.shape[0]
        im_pad_size = 1500
        image_expanded = cv2.copyMakeBorder(image, im_pad_size, im_pad_size, im_pad_size, im_pad_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        masked_image = np.copy(image_expanded)
        class_masked_image = np.ones(masked_image.shape[0:2]) * 255
        masked_image_rgba = np.zeros((masked_image.shape[0], masked_image.shape[1], 4), dtype=np.uint8)
        masked_image_rgba[..., :3] = masked_image

        self._get_bbox_area(_bbox, im_size_X, im_size_Y)
        self._process_annotation(_bbox, image_expanded, masked_image_rgba, class_masked_image, im_pad_size)

        return masked_image_rgba[im_pad_size:-im_pad_size, im_pad_size:-im_pad_size, :], class_masked_image[im_pad_size:-im_pad_size, im_pad_size:-im_pad_size]

    def _get_bbox_area(self, _bbox: dict, im_size_X: int, im_size_Y: int) -> None:
        """
        Update the bounding box dictionary with the area of the bounding box.

        Parameters:
            _bbox (dict): Dictionary containing bounding box coordinates.
            im_size_X (int): Width of the image.
            im_size_Y (int): Height of the image.

        Returns:
            None
        """
        _bbox['bbox_area'] = _bbox['width'] * _bbox['height']

    def _calculate_padded_bbox(self, _bbox: dict, im_pad_size: int) -> np.ndarray:
        """
        Calculate the coordinates of the bounding box after adding padding.

        Parameters:
            _bbox (dict): Dictionary containing bounding box coordinates.
            im_pad_size (int): The amount of padding added to the image.

        Returns:
            np.ndarray: Array containing the padded bounding box coordinates.
        """
        x_min, y_min, x_max, y_max = _bbox["x_min"], _bbox["y_min"], _bbox["x_max"], _bbox["y_max"]
        return np.array([x_min + im_pad_size, y_min + im_pad_size, x_max + im_pad_size, y_max + im_pad_size])
        
    def _process_annotation(self, _bbox: dict, image_expanded: np.ndarray, masked_image_rgba: np.ndarray, class_masked_image: np.ndarray, im_pad_size: int) -> None:
        """
        Processes a single annotation to update the mask images using the SAM predictor.

        Parameters:
            _bbox (dict): Dictionary containing bounding box coordinates and other annotations.
            image_expanded (np.ndarray): The expanded image with padding.
            masked_image_rgba (np.ndarray): The masked image with RGBA channels.
            class_masked_image (np.ndarray): The class mask image (binary mask).
            im_pad_size (int): The padding size added to the image.

        Returns:
            None
        """
        plant_bbox = np.array([int(_bbox['x_min']), int(_bbox['y_min']), int(_bbox['x_max']), int(_bbox['y_max'])])
        sam_crop_size_x, sam_crop_size_y = self._determine_crop_size(_bbox)
        cropped_image = self._crop_image_padded(image_expanded, _bbox, im_pad_size, sam_crop_size_x, sam_crop_size_y)

        self.mask_predictor.set_image(cropped_image)
        log.info(f"Cropped image size for SAM predictor: {cropped_image.shape} ({cropped_image.dtype})")

        _, cropped_bbox = self._get_bounding_boxes(_bbox, plant_bbox, im_pad_size, sam_crop_size_x, sam_crop_size_y)
        input_box = torch.tensor(cropped_bbox, device=self.mask_predictor.device)
        transformed_box = self.mask_predictor.transform.apply_boxes_torch(input_box, cropped_image.shape[:2])

        mask, _, _ = self.mask_predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_box, multimask_output=True, hq_token_only=False)

        self._apply_masks(mask, masked_image_rgba, class_masked_image, _bbox, im_pad_size, sam_crop_size_x, sam_crop_size_y)

    def _get_bounding_boxes(self, _bbox: dict, plant_bbox: np.ndarray, im_pad_size: int, sam_crop_size_x: int, sam_crop_size_y: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the bounding boxes with padding and cropping adjustments.

        Parameters:
            _bbox (dict): Dictionary containing bounding box coordinates and other annotations.
            plant_bbox (np.ndarray): The bounding box of the plant.
            im_pad_size (int): The padding size added to the image.
            sam_crop_size_x (int): The width of the crop size for SAM.
            sam_crop_size_y (int): The height of the crop size for SAM.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Padded bounding box and cropped bounding box arrays.
        """
        padded_bbox = plant_bbox + [im_pad_size, im_pad_size, im_pad_size, im_pad_size]
        cropped_bbox = padded_bbox - [_bbox['center_x'] + im_pad_size - sam_crop_size_x / 2, _bbox['center_y'] + im_pad_size - sam_crop_size_y / 2, _bbox['center_x'] + im_pad_size - sam_crop_size_x / 2, _bbox['center_y'] + im_pad_size - sam_crop_size_y / 2]
        return padded_bbox, cropped_bbox

    def _crop_image(self, image: np.ndarray, y_min: int, y_max: int, x_min: int, x_max: int) -> np.ndarray:
        """
        Crops the image based on the given coordinates.

        Parameters:
            image (np.ndarray): The image to crop.
            y_min (int): The minimum y-coordinate for cropping.
            y_max (int): The maximum y-coordinate for cropping.
            x_min (int): The minimum x-coordinate for cropping.
            x_max (int): The maximum x-coordinate for cropping.

        Returns:
            np.ndarray: The cropped image.
        """
        return image[y_min: y_max, x_min: x_max]

    def _crop_image_padded(self, image_expanded: np.ndarray, _bbox: dict, im_pad_size: int, sam_crop_size_x: int, sam_crop_size_y: int) -> np.ndarray:
        """
        Crops the padded image based on the annotation and padding size.

        Parameters:
            image_expanded (np.ndarray): The expanded image with padding.
            _bbox (dict): Dictionary containing bounding box coordinates and other annotations.
            im_pad_size (int): The padding size added to the image.
            sam_crop_size_x (int): The width of the crop size for SAM.
            sam_crop_size_y (int): The height of the crop size for SAM.

        Returns:
            np.ndarray: The cropped image with padding.
        """
        return np.copy(image_expanded[int(_bbox['center_y'] + im_pad_size - sam_crop_size_y / 2):int(_bbox['center_y'] + im_pad_size + sam_crop_size_y / 2), int(_bbox['center_x'] + im_pad_size - sam_crop_size_x / 2):int(_bbox['center_x'] + im_pad_size + sam_crop_size_x / 2), :])

    def _determine_crop_size(self, _bbox: dict) -> Tuple[int, int]:
        """
        Determines the appropriate crop size based on the dimensions of the bounding box.

        Parameters:
            _bbox (dict): Dictionary containing bounding box dimensions.

        Returns:
            Tuple[int, int]: The width and height of the crop size.
        """
        sam_crop_size_x, sam_crop_size_y = 1000, 1000
        if _bbox['width'] > 700:
            sam_crop_size_x = math.ceil(_bbox['width'] * 1.43 / 2.) * 2
        if _bbox['height'] > 700:
            sam_crop_size_y = math.ceil(_bbox['height'] * 1.43 / 2.) * 2
        return sam_crop_size_x, sam_crop_size_y

    def make_exg(self, rgb_image: np.ndarray, normalize: bool = False, thresh: int = 0) -> np.ndarray:
        """
        Calculates the excess green index (ExG) for an RGB image and applies a threshold if specified.

        Parameters:
            rgb_image (np.ndarray): The input RGB image.
            normalize (bool, optional): Whether to normalize the ExG values. Default is False.
            thresh (int, optional): The threshold value to apply. Pixels with ExG values below this will be set to 0. Default is 0.

        Returns:
            np.ndarray: The computed ExG image.
        """
        rgb_image = rgb_image.astype(float)
        r, g, b = cv2.split(rgb_image)

        if normalize:
            total = r + g + b
            # Avoid division by zero by setting zero total values to 1 (effectively ignoring these pixels)
            total[total == 0] = 1
            exg = 2 * (g / total) - (r / total) - (b / total)
        else:
            exg = 2 * g - r - b

        if thresh is not None and not normalize:
            exg = np.where(exg < thresh, 0, exg)

        return exg

    def get_hsv_from_bgr(self, bgr_list: Tuple[int, int, int]) -> np.ndarray:
        """
        Converts a BGR color value to its corresponding HSV color value.

        Parameters:
            bgr_list (Tuple[int, int, int]): A tuple containing BGR color values.

        Returns:
            np.ndarray: The HSV color value corresponding to the given BGR values.
        """
        bgr_gray = np.array(bgr_list, dtype=np.uint8)
        hsv_gray = cv2.cvtColor(np.uint8([[bgr_gray]]), cv2.COLOR_BGR2HSV)[0][0]
        return hsv_gray

    def remove_gray_hsv_color(self, hsv_image: np.ndarray) -> np.ndarray:
        """
        Removes gray colors from an HSV image by creating a mask that filters out gray hues.

        Parameters:
            hsv_image (np.ndarray): The input image in HSV color space.

        Returns:
            np.ndarray: A binary mask where gray colors are removed (set to 0), and all other colors are kept (set to 255).
        """
        # Define the lower and upper bounds for gray colors in HSV.
        lower_gray = (0, 0, 50) # Lower hsv for gray color
        upper_gray = (180, 60, 200) # Upper hsv for gray color 

        # Create a mask for the gray color range in HSV
        mask = cv2.inRange(hsv_image, np.array(lower_gray), np.array(upper_gray))
        
        # Invert the mask to exclude gray colors
        mask_gray = cv2.bitwise_not(mask)
        
        return mask_gray     

    def _clean_mask(self, mask: np.ndarray, cropped_image_area: np.ndarray, image_id: str, class_id: str) -> np.ndarray:
        """
        Cleans up the mask using morphological operations and filtering techniques. Removes gray colors, and applies 
        specific post-processing based on the class ID for fine-tuning the mask.

        Parameters:
            mask (np.ndarray): Binary mask to be cleaned.
            cropped_image_area (np.ndarray): Cropped image area where the mask is applied.
            image_id (str): Identifier for the image (not used in this method but can be used for logging or debugging).
            class_id (str): Identifier for the class of the object to determine the type of morphology to apply.

        Returns:
            np.ndarray: The cleaned mask after applying morphological operations and filtering.
        """
        log.info("Starting clean mask.")

        # Broadcast the mask to 3 channels to match the image dimensions
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        # Apply the mask to the cropped image area
        cutout = np.where(mask_3d == 1, cropped_image_area, 0)

        cutout_bgr = cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB)
        cutout_hsv = cv2.cvtColor(cutout_bgr, cv2.COLOR_RGB2HSV)
        cutout_mask_gray = self.remove_gray_hsv_color(cutout_hsv).astype(np.uint8)  # Remove gray background

        combined_cutout_mask = np.where(cutout_mask_gray == 255, mask, 0)
        cutout_mask_gray_3d = np.repeat(cutout_mask_gray[:, :, np.newaxis], 3, axis=2)

        # Remove gray areas from the cutout
        cutout_gray_removed_bgr = np.where(cutout_mask_gray_3d == 255, cutout, 1)
        cutout_gray_removed_rgb = cv2.cvtColor(cutout_gray_removed_bgr, cv2.COLOR_BGR2RGB)

        # Apply post-processing based on class ID
        if class_id in self.broad_spceies:
            log.info(f"Working with broad morphology, class_id: {class_id}")
            cleaned_mask = self._clean_sparse(cutout_gray_removed_rgb)
        elif class_id in self.sparse_spceies:
            log.info(f"Working with sparse morphology, class_id: {class_id}")
            cleaned_mask = self._clean_broad(class_id, combined_cutout_mask)
        else:
            log.error(f"class_id: {class_id} not defined in broad_sprase_morph_species")
            cleaned_mask = np.zeros_like(mask)  # Return an empty mask if class_id is not defined

        return cleaned_mask

    def _clean_sparse(self, cutout_gray_removed_rgb: np.ndarray) -> np.ndarray:
        """
        Cleans up the mask for sparse morphology using ExG (Excess Green) filtering and morphological operations.

        Parameters:
            cutout_gray_removed_rgb (np.ndarray): Image with gray colors removed and converted to RGB.

        Returns:
            np.ndarray: The cleaned mask after applying ExG filtering and morphological operations.
        """
        # Calculate ExG (Excess Green) index and create a mask
        exg_image = self.make_exg(cutout_gray_removed_rgb)
        exg_mask = np.where(exg_image > 0, 1, 0).astype(np.uint8)
        
        # Broadcast the mask to match the image dimensions
        exg_mask_3d = np.repeat(exg_mask[:, :, np.newaxis], 3, axis=2)
        # Apply the ExG mask to the image
        cutout_exg = np.where(exg_mask_3d == 1, cutout_gray_removed_rgb, 0)

        # Apply morphological operations to clean up the mask
        cleaned_mask = remove_small_holes(exg_mask.astype(bool), area_threshold=100, connectivity=2).astype(np.uint8)
        cleaned_mask = remove_small_objects(cleaned_mask.astype(bool), min_size=100, connectivity=2).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.GaussianBlur(cleaned_mask, (7, 7), sigmaX=1)
        return cleaned_mask

    def _clean_broad(self, class_id: str, combined_cutout_mask: np.ndarray) -> np.ndarray:
        """
        Cleans up the mask for broad morphology using morphological operations.

        Parameters:
            class_id (str): Identifier for the class of the object (not used directly but for context).
            combined_cutout_mask (np.ndarray): Combined mask to be cleaned.

        Returns:
            np.ndarray: The cleaned mask after applying morphological operations.
        """
        # Apply morphological operations to clean up the mask
        cleaned_mask = remove_small_holes(combined_cutout_mask.astype(bool), area_threshold=100).astype(np.uint8)
        cleaned_mask = remove_small_objects(cleaned_mask.astype(bool), min_size=100, connectivity=2).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.GaussianBlur(cleaned_mask, (7, 7), sigmaX=1)
        return cleaned_mask

    def _apply_masks(self, masks: torch.Tensor, masked_image_rgba: np.ndarray, class_masked_image: np.ndarray, _bbox: dict, im_pad_size: int, sam_crop_size_x: int, sam_crop_size_y: int) -> None:
        """
        Applies generated masks to the respective images and updates the mask images. Utilizes _clean_mask to refine the masks.

        Parameters:
            masks (torch.Tensor): Tensor of masks predicted by the model.
            masked_image_rgba (np.ndarray): Image with initial masking applied.
            class_masked_image (np.ndarray): Image to store class-specific mask information.
            _bbox (dict): Bounding box information including coordinates and image ID.
            im_pad_size (int): Padding size used for cropping.
            sam_crop_size_x (int): Width of the cropped region.
            sam_crop_size_y (int): Height of the cropped region.

        Returns:
            None
        """
        bb_color = tuple(np.random.random(size=3) * 255)  # Random color for bounding box visualization

        for mask in masks:
            # Create an empty mask for the full image
            full_mask = np.zeros(masked_image_rgba.shape[0:2])

            # Define cropping coordinates
            crop_start_y = int(_bbox['center_y'] + im_pad_size - sam_crop_size_y / 2)
            crop_end_y = int(_bbox['center_y'] + im_pad_size + sam_crop_size_y / 2)
            crop_start_x = int(_bbox['center_x'] + im_pad_size - sam_crop_size_x / 2)
            crop_end_x = int(_bbox['center_x'] + im_pad_size + sam_crop_size_x / 2)

            # Apply the mask to the cropped region
            full_mask[crop_start_y:crop_end_y, crop_start_x:crop_end_x] = mask.cpu()[0, :, :]

            cropped_image_area = masked_image_rgba[crop_start_y:crop_end_y, crop_start_x:crop_end_x, :3]
            cropped_mask = full_mask[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

            # Clean the cropped mask
            cleaned_cropped_mask = self._clean_mask(cropped_mask, cropped_image_area, _bbox["image_id"], _bbox["class_id"])
            full_mask[crop_start_y:crop_end_y, crop_start_x:crop_end_x] = cleaned_cropped_mask

            # Update the image with the cleaned mask
            alpha = 0.5
            for c in range(3):
                masked_image_rgba[full_mask == 1, c] = (1 - alpha) * masked_image_rgba[full_mask == 1, c] + alpha * bb_color[c]
            masked_image_rgba[full_mask == 1, 3] = int(alpha * 255)
            class_masked_image[full_mask == 1] = _bbox['class_id']

class DirectoryInitializer:
    def __init__(self, cfg: DictConfig) -> None:
        """
        The DirectoryInitializer class initializes

        Parameters:
            cfg (DictConfig): The configuration object.

        Returns:
            None
        """
        self.image_dir = Path(cfg.data.temp_image_dir)
        self.metadata_dir = Path(cfg.data.image_metadata_dir)
        self.output_dir = Path(cfg.data.temp_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_images(self) -> List[Path]:
        """
        Get the list of images from the image directory.

        Returns:
            List[Path]: List of image paths.
        """
        return sorted([x for x in self.image_dir.glob("*.JPG")])

def process_sequentially(directory_initializer: DirectoryInitializer, processor: SingleImageProcessor) -> None:
    """
    This function processes a batch of images sequentially.

    Parameters:
        directory_initializer (DirectoryInitializer): The DirectoryInitializer object.
        processor (SingleImageProcessor): The SingleImageProcessor object.

    Returns:
        None
    """
    imgs = directory_initializer.get_images()
    for image_path in imgs:
        log.info(f"Processing image: {image_path}")
        json_path = str(directory_initializer.metadata_dir / f"{image_path.stem}.json")
        input_paths = (image_path, json_path)
        # processor.process_image(input_paths)
        processor.save_cutout(input_paths)

def process_concurrently(directory_initializer: DirectoryInitializer, processor: SingleImageProcessor) -> None:
    """
    This function processes a batch of images concurrently using multiprocessing.

    Parameters:
        directory_initializer (DirectoryInitializer): The DirectoryInitializer object.
        processor (SingleImageProcessor): The SingleImageProcessor object.

    Returns:
        None
    """
    imgs = directory_initializer.get_images()
    input_paths = [
        (str(imgpath), str(directory_initializer.metadata_dir / f"{imgpath.stem}.json"))
        for imgpath in imgs
    ]
    
    max_workers = int(len(os.sched_getaffinity(0)) / 5)
    log.info(f"Using {max_workers} workers for multiprocessing")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as executor:
        # executor.map(processor.process_image, input_paths)
        executor.map(processor.save_cutout, input_paths)

def main(cfg: DictConfig) -> None:
    """
    The main function to process a batch of images using the SingleImageProcessor.
    
    Parameters:
        cfg (DictConfig): The configuration object.

    Returns:
        None
    """
    log.info("Starting segmentation.")
    directory_initializer = DirectoryInitializer(cfg)
    imgs = directory_initializer.get_images()
    output_dir = directory_initializer.output_dir
    metadata_dir = directory_initializer.metadata_dir

    processor = SingleImageProcessor(
        cfg=cfg,
        output_dir=output_dir,
        metadata_dir=metadata_dir,
        model_type=cfg.data.sam_model_type,
        sam_checkpoint=cfg.data.sam_hq_checkpoint
    )

    multiprocess = False
    if multiprocess:
        log.info("Starting concurrent processing")
        process_concurrently(directory_initializer, processor)
    else:
        log.info("Starting sequential processing")
        process_sequentially(directory_initializer, processor)
