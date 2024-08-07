import os
import json
import math
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
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
    def __init__(self, output_dir: str, metadata_dir: str, model_type: str, sam_checkpoint: str):
        """
        Initializes the SingleImageProcessor with image and JSON paths, and model details.
        """
        log.info("Initializing SingleImageProcessor")
        self.metadata_dir = Path(metadata_dir)
        self.output_dir = Path(output_dir)
        self.visualization_label_dir = self.output_dir / "vis_masks" # before _clean_mask???????? 

        for output_dir in [self.output_dir, self.visualization_label_dir]:
            output_dir.mkdir(exist_ok=True, parents=True)

        log.info("Loading SAM model")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.mask_predictor = SamPredictor(sam)

    def process_image(self, input_paths: Tuple[str, str]):
        """
        Processes a single image and its corresponding annotations from JSON.
        """
        image_path, json_path = input_paths
        log.info(f"Processing image: {image_path}")

        if Path(json_path).exists():
            with open(json_path, 'r') as f:
                data = json.load(f)

            bbox = data['detection_results']['bbox']
            bbox.update({
                'image_id': data['detection_results']["image_id"],
                'class_id': data['detection_results']["class_id"],
                'width': bbox['x_max'] - bbox['x_min'],
                'height': bbox['y_max'] - bbox['y_min']
            })

            log.info(f"Extracted bounding box: {bbox}")
            self._find_bbox_center(bbox)
            image = self._read_image(image_path)
            masked_image, class_masked_image = self._create_masks(image, bbox)

            class_id = data['detection_results']["class_id"]

            save_class_dir = Path(self.output_dir, f"{class_id}")
            save_class_dir.mkdir(exist_ok=True, parents=True)

            mask_name = Path(image_path).stem + '.png'
            log.info(f"Saving masks ({Path(image_path).stem}) to {self.visualization_label_dir.parent}")
            self.save_compressed_image(masked_image, self.visualization_label_dir / mask_name) # masked image saved in visualization_label_dir
            cv2.imwrite(str(save_class_dir / mask_name), class_masked_image.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 1]) # class_masked_image saved in output_dir
            class_masked_image_3d = np.repeat(class_masked_image[:, :, np.newaxis], 3, axis=2).astype(np.uint8) # convert the mask to 3d

            # Save the final cutout:
            mask_cutout_name = Path(image_path).stem + '_cutout.png'
            
            class_masked_image_3d = np.repeat(class_masked_image[:, :, np.newaxis], 3, axis=2).astype(np.uint8) # convert the mask to 3d
            mask_cutout_bgr = np.where(class_masked_image_3d != 255, image, 0)
            mask_cutout_hsv = cv2.cvtColor(mask_cutout_bgr, cv2.COLOR_BGR2HSV)
            mask_gray_removed = self.remove_gray_hsv_color(mask_cutout_hsv)
            mask_cutout_bgr = cv2.bitwise_and(mask_cutout_bgr, mask_cutout_bgr, mask=mask_gray_removed)
            final_mask_cutout = cv2.cvtColor(mask_cutout_bgr, cv2.COLOR_BGR2RGB)

            log.info("Saving mask cutout")
            cv2.imwrite(str(save_class_dir / mask_cutout_name), final_mask_cutout.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 1])  # mask_cutout saved in output_dir
        else:
            log.error(f"No JSON file for {json_path}")
            return None
        

    def save_compressed_image(self, image: np.ndarray, path: str, quality: int = 98):
        """
        Save the image in a compressed format.
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        is_success, encoded_image = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if is_success:
            with open(path, 'wb') as f:
                encoded_image.tofile(f)

    def _find_bbox_center(self, bbox: dict):
        """
        Processes annotation data to calculate bounding box coordinates.
        """
        bbox['center_x'] = (bbox['x_min'] + bbox['x_max']) / 2
        bbox['center_y'] = (bbox['y_min'] + bbox['y_max']) / 2

    def _read_image(self, image_path: Path) -> np.ndarray:
        """
        Reads an image from the specified path.
        """
        log.info(f"Reading image and converting to RGB: {image_path}")
        return cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

    def _create_masks(self, image: np.ndarray, bbox: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates masked images based on annotations.
        """
        im_size_X, im_size_Y = image.shape[1], image.shape[0]
        im_pad_size = 1500
        image_expanded = cv2.copyMakeBorder(image, im_pad_size, im_pad_size, im_pad_size, im_pad_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        masked_image = np.copy(image_expanded)
        class_masked_image = np.ones(masked_image.shape[0:2]) * 255
        masked_image_rgba = np.zeros((masked_image.shape[0], masked_image.shape[1], 4), dtype=np.uint8)
        masked_image_rgba[..., :3] = masked_image

        self._get_bbox_area(bbox, im_size_X, im_size_Y)
        self._process_annotation(bbox, image_expanded, masked_image_rgba, class_masked_image, im_pad_size)

        return masked_image_rgba[im_pad_size:-im_pad_size, im_pad_size:-im_pad_size, :], class_masked_image[im_pad_size:-im_pad_size, im_pad_size:-im_pad_size]

    def _get_bbox_area(self, bbox: dict, im_size_X: int, im_size_Y: int):
        """
        Updates annotations with absolute coordinates based on image dimensions.
        """
        bbox['bbox_area'] = bbox['width'] * bbox['height']

    def _calculate_padded_bbox(self, bbox: dict, im_pad_size: int) -> np.ndarray:
        """
        Calculate the padded bounding box coordinates.
        """
        x_min, y_min, x_max, y_max = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
        return np.array([x_min + im_pad_size, y_min + im_pad_size, x_max + im_pad_size, y_max + im_pad_size])

    def _process_annotation(self, bbox: dict, image_expanded: np.ndarray, masked_image_rgba: np.ndarray, class_masked_image: np.ndarray, im_pad_size: int):
        """
        Processes a single annotation and updates the mask images. Sam is executed here.
        """
        plant_bbox = np.array([int(bbox['x_min']), int(bbox['y_min']), int(bbox['x_max']), int(bbox['y_max'])])
        sam_crop_size_x, sam_crop_size_y = self._determine_crop_size(bbox)
        cropped_image = self._crop_image(image_expanded, bbox, im_pad_size, sam_crop_size_x, sam_crop_size_y)

        self.mask_predictor.set_image(cropped_image)
        log.info(f"Cropped image size for SAM predictor: {cropped_image.shape} ({cropped_image.dtype})")

        _, cropped_bbox = self._get_bounding_boxes(bbox, plant_bbox, im_pad_size, sam_crop_size_x, sam_crop_size_y)
        input_box = torch.tensor(cropped_bbox, device=self.mask_predictor.device)
        transformed_box = self.mask_predictor.transform.apply_boxes_torch(input_box, cropped_image.shape[:2])

        mask, _, _ = self.mask_predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_box, multimask_output=True, hq_token_only=False)

        
        self._apply_masks(mask, masked_image_rgba, class_masked_image, bbox, im_pad_size, sam_crop_size_x, sam_crop_size_y)

    def _get_bounding_boxes(self, bbox: dict, plant_bbox: np.ndarray, im_pad_size: int, sam_crop_size_x: int, sam_crop_size_y: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the padded and cropped bounding boxes.
        """
        padded_bbox = plant_bbox + [im_pad_size, im_pad_size, im_pad_size, im_pad_size]
        cropped_bbox = padded_bbox - [bbox['center_x'] + im_pad_size - sam_crop_size_x / 2, bbox['center_y'] + im_pad_size - sam_crop_size_y / 2, bbox['center_x'] + im_pad_size - sam_crop_size_x / 2, bbox['center_y'] + im_pad_size - sam_crop_size_y / 2]
        return padded_bbox, cropped_bbox

    def _crop_image(self, image_expanded: np.ndarray, bbox: dict, im_pad_size: int, sam_crop_size_x: int, sam_crop_size_y: int) -> np.ndarray:
        """
        Crops the image based on the annotation and padding size.
        """
        return np.copy(image_expanded[int(bbox['center_y'] + im_pad_size - sam_crop_size_y / 2):int(bbox['center_y'] + im_pad_size + sam_crop_size_y / 2), int(bbox['center_x'] + im_pad_size - sam_crop_size_x / 2):int(bbox['center_x'] + im_pad_size + sam_crop_size_x / 2), :])

    def _determine_crop_size(self, bbox: dict) -> Tuple[int, int]:
        """
        Determines the appropriate crop size based on annotation dimensions.
        """
        sam_crop_size_x, sam_crop_size_y = 1000, 1000
        if bbox['width'] > 700:
            sam_crop_size_x = math.ceil(bbox['width'] * 1.43 / 2.) * 2
        if bbox['height'] > 700:
            sam_crop_size_y = math.ceil(bbox['height'] * 1.43 / 2.) * 2
        return sam_crop_size_x, sam_crop_size_y

    def make_exg(self, rgb_image: np.ndarray, normalize: bool = False, thresh: int = 0) -> np.ndarray:
        """
        Calculate the excess green index (ExG) for an RGB image and apply a threshold.
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

        return exg#.astype("uint8")
    
    
    def get_hsv_from_bgr(self, bgr_list):
        bgr_gray = np.array(bgr_list, dtype=np.uint8)

        # Convert BGR gray color to HSV
        hsv_gray = cv2.cvtColor(np.uint8([[bgr_gray]]), cv2.COLOR_BGR2HSV)[0][0]
        return hsv_gray


    def remove_gray_hsv_color(self, hsv_image) -> np.ndarray:
        
        lower_gray=(0, 0, 50)
        upper_gray=(180, 50, 200)

        # Create a mask for the gray color
        mask = cv2.inRange(hsv_image, np.array(lower_gray), np.array(upper_gray))

        # Invert the mask to get the mask for all colors except gray
        mask_inv = cv2.bitwise_not(mask)

        # # Use the inverted mask to keep only the non-gray areas of the original image
        # result = cv2.bitwise_and(image, image, mask=mask_inv)

        # Invert the mask to keep everything except the gray color
        mask_gray = cv2.bitwise_not(mask)
        return mask_gray


    def _clean_mask(self, mask: np.ndarray, cropped_image_area: np.ndarray, image_id: str, class_id: str) -> np.ndarray:
        # TODO: adjust this for different species
        """
        Applies morphological opening and closing operations to clean up the mask,
        removes large non-green components.
        """
        log.info("Starting clean mask.")
        # Broadcast the mask to have the same shape as the image
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        # Apply the mask to the image
        cutout = np.where(mask_3d == 1, cropped_image_area, 0)

        # Apply different masks based on different morphology
        sparse_morphology = [2, 5,6, 7, 8, 19, 11, 12, 15, 20, 23, 24, 42] # only ragweed. incluse grasses.
        broad_morphology = [1, 3, 4, 9, 14, 16, 18, 21, 22, 25, 43, 44, 45, 46] #test and see what works

        if class_id in sparse_morphology:
            # Apply exg (this is good for ragweed parthenium but not for cocklebur)
            log.info("Working with sparse morphology")
            exg_image = self.make_exg(cutout)
            exg_mask = np.where(exg_image > 0, 1, 0).astype(np.uint8)
            # Apply morphological closing and opening
            cleaned_mask = remove_small_holes(exg_mask.astype(bool), area_threshold=100, connectivity=2).astype(np.uint8)
            cleaned_mask = remove_small_objects(cleaned_mask.astype(bool), min_size=100, connectivity=2).astype(np.uint8)

            cleaned_mask = cv2.GaussianBlur(cleaned_mask, (7, 7), sigmaX=1)
            kernel = np.ones((3, 3), np.uint8)  # Default kernel size, assuming 5x5
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
            
            cleaned_mask = remove_small_holes(cleaned_mask.astype(bool), area_threshold=10).astype(np.uint8)
            # blurred_mask = cv2.GaussianBlur(cleaned_mask.astype(np.uint8), gaussian_blur_size, 0) # guassian blur to smooth edges
            # _, cleaned_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY) # convert blurred mask back to a binary mask by thresholding

            # Apply the mask to the image
            # mask_3d = np.repeat(cleaned_mask[:, :, np.newaxis], 3, axis=2)
            # exged_cutout = np.where(mask_3d > 0, cutout, 0)
            # temp_image_dir = f"data/images_testing/temp_exged_cutout_cleaned/"
            # Path(temp_image_dir).mkdir(exist_ok=True, parents=True)
            # temp_image_path = str(Path(temp_image_dir, f"{image_id}.png"))
            
            # bgr_result = cv2.cvtColor(exged_cutout, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(temp_image_path ,bgr_result)
            
        elif class_id in broad_morphology:
            log.info("Working with broad morphology")
            # Apply morphological closing and opening
            cutout_gray = cv2.cvtColor(cutout, cv2.COLOR_RGB2GRAY).astype(np.uint8)
            cleaned_mask = remove_small_holes(cutout_gray.astype(bool), area_threshold=1).astype(np.uint8)
            # cleaned_mask = cv2.GaussianBlur(cleaned_mask, (7, 7), sigmaX=1)
            # kernel = np.ones((5, 5), np.uint8)  # Default kernel size, assuming     5x5
            # cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
            # cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
            
            # blurred_mask = cv2.GaussianBlur(cleaned_mask.astype(np.uint8), gaussian_blur_size, 0) # guassian blur to smooth edges
            # _, cleaned_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY) # convert blurred mask back to a binary mask by thresholding


        # Apply the mask to the image
        # mask_3d = np.repeat(cleaned_mask[:, :, np.newaxis], 3, axis=2)
        # exged_cutout = np.where(mask_3d > 0, cutout, 0)
        # temp_image_dir = f"data/images_testing/temp_exged_cutout_cleaned/"
        # Path(temp_image_dir).mkdir(exist_ok=True, parents=True)
        # temp_image_path = str(Path(temp_image_dir, f"{image_id}.png"))
        
        # bgr_result = cv2.cvtColor(exged_cutout, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(temp_image_path ,bgr_result)

        else:
            log.info("Working with unknown morphology")
            # Apply morphological closing and opening
            cutout_gray = cv2.cvtColor(cutout, cv2.COLOR_RGB2GRAY)
            cleaned_mask = cv2.GaussianBlur(cutout_gray, (7, 7), sigmaX=1)
            kernel = np.ones((3, 3), np.uint8)  # Default kernel size, assuming 5x5
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
            cleaned_mask = remove_small_holes(cleaned_mask.astype(bool), area_threshold=10).astype(np.uint8)
            # blurred_mask = cv2.GaussianBlur(cleaned_mask.astype(np.uint8), gaussian_blur_size, 0) # guassian blur to smooth edges
            # _, cleaned_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY) # convert blurred mask back to a binary mask by thresholding

        return cleaned_mask
        
    def _apply_masks(self, masks: torch.Tensor, masked_image_rgba: np.ndarray, class_masked_image: np.ndarray, bbox: dict, im_pad_size: int, sam_crop_size_x: int, sam_crop_size_y: int):
        """
        Applies the generated masks to the respective mask images. _clean_mask is executed here.
        """
        bb_color = tuple(np.random.random(size=3) * 255)
        for mask in masks:
            full_mask = np.zeros(masked_image_rgba.shape[0:2])
            crop_start_y = int(bbox['center_y'] + im_pad_size - sam_crop_size_y / 2)
            crop_end_y = int(bbox['center_y'] + im_pad_size + sam_crop_size_y / 2)
            crop_start_x = int(bbox['center_x'] + im_pad_size - sam_crop_size_x / 2)
            crop_end_x = int(bbox['center_x'] + im_pad_size + sam_crop_size_x / 2)
            full_mask[crop_start_y:crop_end_y, crop_start_x:crop_end_x] = mask.cpu()[0, :, :]

            cropped_image_area = masked_image_rgba[crop_start_y:crop_end_y, crop_start_x:crop_end_x, :3]
            cropped_mask = full_mask[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

            cleaned_cropped_mask = self._clean_mask(cropped_mask, cropped_image_area, bbox["image_id"], bbox["class_id"])
            # cleaned_cropped_mask = cropped_mask
            full_mask[crop_start_y:crop_end_y, crop_start_x:crop_end_x] = cleaned_cropped_mask # applied cleaned mask (from _clean_mask to the full mask) 
            alpha = 0.5
            for c in range(3):
                masked_image_rgba[full_mask == 1, c] = (1 - alpha) * masked_image_rgba[full_mask == 1, c] + alpha * bb_color[c]
            masked_image_rgba[full_mask == 1, 3] = int(alpha * 255)
            class_masked_image[full_mask == 1] = bbox['class_id']

class DirectoryInitializer:
    def __init__(self, cfg: DictConfig):
        """
        Initialize directory structure based on the configuration.
        """
        self.image_dir = Path(cfg.data.temp_image_dir)
        self.metadata_dir = Path(cfg.data.image_metadata_dir)
        self.output_dir = Path(cfg.data.temp_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_images(self) -> List[Path]:
        """
        Get a list of image paths from the image directory.
        """
        return sorted([x for x in self.image_dir.glob("*.JPG")])

def process_sequentially(directory_initializer: DirectoryInitializer, processor: SingleImageProcessor):
    """
    Process a batch of images sequentially.
    """
    imgs = directory_initializer.get_images()
    for image_path in imgs:
        log.info(f"Processing image: {image_path}")
        json_path = str(directory_initializer.metadata_dir / f"{image_path.stem}.json")
        input_paths = (image_path, json_path)
        processor.process_image(input_paths)

def process_concurrently(directory_initializer: DirectoryInitializer, processor: SingleImageProcessor):
    """
    Process a batch of images using multiprocessing.
    """
    imgs = directory_initializer.get_images()
    input_paths = [
        (str(imgpath), str(directory_initializer.metadata_dir / f"{imgpath.stem}.json"))
        for imgpath in imgs
    ]
    
    max_workers = int(len(os.sched_getaffinity(0)) / 5)
    log.info(f"Using {max_workers} workers for multiprocessing")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as executor:
        executor.map(processor.process_image, input_paths)

def main(cfg: DictConfig) -> None:
    """Main function to process a batch of images, either sequentially or concurrently."""
    log.info("Starting batch processing")
    directory_initializer = DirectoryInitializer(cfg)
    imgs = directory_initializer.get_images()
    output_dir = directory_initializer.output_dir
    metadata_dir = directory_initializer.metadata_dir

    processor = SingleImageProcessor(
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
