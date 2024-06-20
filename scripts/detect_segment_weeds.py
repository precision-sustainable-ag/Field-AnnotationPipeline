import json
import logging
from pathlib import Path


import cv2
import numpy as np
from omegaconf import DictConfig
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class SegmentWeeds:
    """
    A class to handle weed detection and segmentation of images from Field-Image Repository.

    Methods:
        __init__(self, cfg: DictConfig): Initializes directories and model paths.
        detect_and_segment(self, image_path): Detects and segments weeds in the given image.
        apply_mask(self, image, mask): Applies a mask to the image with transparency.
        create_cutout(self, image, mask): Creates a cutout image based on the mask.
        save_image(self, image, bbox, path): Saves the image to the specified path, optionally with a bounding box.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes directories and model paths.

        Parameters:
            cfg (DictConfig): Configuration object containing directory paths and model details.
        """
        # Define directories and model paths
        self.image_dir = Path(cfg.data.temp_image_dir)
        self.output_dir = Path(cfg.data.temp_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.path_yolo_model = cfg.data.path_yolo_model
        self.sam_checkpoint = cfg.data.sam_checkpoint
        self.sam_model_type = cfg.data.sam_model_type

    def detect_and_segment(self, image_path):
        """
        Detects and segments weeds in the given image.

        Parameters:
            image_path (Path): Path to the image file to be processed.
        """
        try:
            # Load pretrained YOLO model and perform detection on the image
            yolo_model = YOLO(self.path_yolo_model)
            results = yolo_model(str(image_path))

            # Extract bounding boxes from YOLO results
            if not results:
                log.warning(f"No detection for image: {image_path}")
                return
            bbox = results[0].boxes.xyxy.tolist()[0]

            # Load and convert the image to RGB format for SAM
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

            # Load the SAM model
            sam = sam_model_registry[self.sam_model_type](
                checkpoint=self.sam_checkpoint
            )
            predictor = SamPredictor(sam)

            # Set the image for the SAM predictor
            predictor.set_image(image)

            # Perform SAM prediction to obtain the mask
            input_box = np.array(bbox)
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

            # Create a segmented image with mask overlay
            segmented_image = self.apply_mask(image, masks[0])
            # Save the segmented image
            segmented_image_name = image_path.stem + "_segmented.JPG"
            segmented_image_path = self.output_dir / segmented_image_name
            self.save_image(segmented_image, input_box, segmented_image_path)

            # Create and save the cutout image
            cutout_image = self.create_cutout(image, masks[0])
            cutout_image_name = image_path.stem + "_cutout.JPG"
            cutout_image_path = self.output_dir / cutout_image_name
            self.save_image(cutout_image, None, cutout_image_path)

        except Exception as e:
            log.error(f"Failed to process {image_path}: {e}", exc_info=True)

    def apply_mask(self, image, mask):
        """
        Applies a mask to the image with transparency.

        Parameters:
            image (np.ndarray): The original image.
            mask (np.ndarray): The mask to be applied.

        Returns:
            np.ndarray: The image with the mask applied.
        """
        mask_color = np.array([30, 144, 255])  # Blue color for the mask
        alpha = 0.6  # Transparency for the mask
        segmented_image = image.copy()
        segmented_image[mask > 0.5] = (
            segmented_image[mask > 0.5] * (1 - alpha) + mask_color * alpha
        ).astype(np.uint8)
        return segmented_image

    @staticmethod
    def load_species_info(path):
        logging.debug(f"Loading species info from {path}.")
        with open(path, "r") as outfile:
            data = json.load(outfile)
        return data

    def fix_missmatched_common_names(weedtype):
        {"Waterhemp": ["common waterhemp", "waterhemp", "waterhemp common"], 
         "Kochia": ["common kochia", "summer cypress"]}

        # from table     # from species_info.json
        {"Common kochia": "BASC5"}
        

    
    def create_mask(mask, fixed_weedtype):
        species_info_path = "/home/psa_images/SemiF-AnnotationPipeline/data/semifield-utils/species_information/species_info.json"
        specie_dict = SegmentWeeds.load_species_info(species_info_path)["species"]

        class_id = specie_dict[fixed_weedtype]

        new_mask = np.where(mask == 1, class_id, 0)
        return new_mask








    def create_cutout(self, image, mask):
        """
        Creates a cutout image based on the mask.

        Parameters:
            image (np.ndarray): The original image.
            mask (np.ndarray): The mask to create the cutout.

        Returns:
            np.ndarray: The cutout image.
        """
        binary_mask = (mask > 0.5).astype(np.uint8)
        cutout_image = cv2.bitwise_and(image, image, mask=binary_mask)
        return cutout_image

    def save_image(self, image, bbox, path):
        """
        Saves the image to the specified path, optionally with a bounding box.

        Parameters:
            image (np.ndarray): The image to be saved.
            bbox (list or None): The bounding box to be drawn on the image.
            path (Path): The path to save the image.
        """
        if bbox is not None:
            cv2.rectangle(
                image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2,
            )
        cv2.imwrite(
            str(path),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        )


def main(cgfg: DictConfig) -> None:m
    """
    Main function to start the weed detection and segmentation task.

    Parameters:
        cfg (DictConfig): Configuration object containing the task details.
    """
    log.info(f"Starting {cfg.general.task}")
    segment_weeds = SegmentWeeds(cfg)

    # Process each image in the directory
    for image_path in segment_weeds.image_dir.iterdir():
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            segment_weeds.detect_and_segment(image_path)
    log.info(f"{cfg.general.task} completed.")
