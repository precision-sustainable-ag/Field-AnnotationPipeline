import cv2
import json
import torch
import logging
import numpy as np

from PIL import Image
from pathlib import Path
from datetime import datetime
from src.utils.unet import UNet
from omegaconf import DictConfig
from torchvision import transforms

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
log = logging.getLogger(__name__)

# Set device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

class UNetInference:
    """
    A class for performing inference using a pre-trained UNet model.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the UNetInference class, load the model, and prepare directories.

        Args:
            cfg (DictConfig): Configuration object containing paths and settings.
        """
        log.info(f"Initializing UNetInference at {datetime.now()}")
        
        # Set up directories
        self.temp_dir = Path(cfg.paths.temp_dir)

        self.trained_model_path = cfg.paths.unet_segmentation_model
        log.info(f"Trained UNet model located at: {self.trained_model_path}")

        # Load UNet model
        log.info("Loading UNet model for segmentation.")
        self.seg_model = UNet(in_channels=3, num_classes=1).to(device)
        self.seg_model.load_state_dict(torch.load(self.trained_model_path, map_location=device))
        self.seg_model.eval()
        log.info("UNet model loaded and set to evaluation mode.")

        # Define image transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _read_image(self, image_path: Path) -> np.ndarray:
        log.info(f"Reading image and converting to RGB: {image_path}")
        return cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

    def _convert_to_rgb(self, image_path: str) -> np.ndarray:
        """
        Converts the given image to RGB format.

        Args:
            image_path (str): Path to the input image.

        Returns:
            np.ndarray: Image as a numpy array in RGB format.
        """
        image = cv2.imread(str(image_path))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _find_bbox_center(self, _bbox: dict) -> None:
        log.info(f"Finding center of the bounding box.")
        _bbox['center_x'] = (_bbox['x_min'] + _bbox['x_max']) / 2
        _bbox['center_y'] = (_bbox['y_min'] + _bbox['y_max']) / 2

    def _crop_image_bbox(self, image_path: str, _bbox: dict):
        """
        Crops the image using the detected bounding box.

        Args:
            image_path (str): Path to the input image.

        Returns:
            tuple: Cropped image, bounding box, and original image.
        """
        image_full_size = self._convert_to_rgb(image_path)
        
        x_min, y_min, x_max, y_max = _bbox["x_min"], _bbox["y_min"], _bbox["x_max"], _bbox["y_max"]
        cropped_image = image_full_size[y_min:y_max, x_min:x_max]

        return cropped_image, _bbox, image_full_size
    
    def _process_image(self, image_path: str, _bbox: dict):
        """Crop the image and return the cropped image, bounding box, and full-size image."""
        cropped_image_bbox, bbox, image_full_size = self._crop_image_bbox(image_path, _bbox)
        return cropped_image_bbox, bbox, image_full_size 

    def _predict_mask(self, cropped_image: np.ndarray):
        """Perform segmentation inference on the cropped image."""
        pil_image = Image.fromarray(cropped_image)
        image_tensor = self.transform(pil_image).float().to(device).unsqueeze(0)

        pred_mask = self.seg_model(image_tensor)
        # Apply sigmoid to convert logits to probabilities (for binary)
        pred_mask = torch.sigmoid(pred_mask)
        
        pred_mask = pred_mask.squeeze(0).cpu().detach().permute(1, 2, 0)
        pred_mask = (pred_mask > 0.5).float().numpy()

        pred_mask = pred_mask.squeeze(-1)

        return pred_mask
    
    def _process_image_in_tiles(self, image: np.ndarray, overlap_pixels=500):
        """Process the image in tiles to avoid memory issues.
        Args:
            image (np.ndarray): The input image.
            overlap_pixels (int): The number of overlapping pixels between tiles.
        Returns:
            np.ndarray: The full-size mask for the input image.
        """
        height, width = image.shape[:2]
        tile_h, tile_w = height // 2, width // 2
        step_h, step_w = tile_h - overlap_pixels, tile_w - overlap_pixels

        pred_mask = np.zeros((height, width), dtype=np.float32)

        for y in range(0, height, step_h):
            for x in range(0, width, step_w):
                y_end, x_end = min(y + tile_h, height), min(x + tile_w, width) # Calculate end coordinates for tile

                tile = image[y:y_end, x:x_end] # Extract tile from image
                tile_pred = self._predict_mask(tile) # Predict mask for tile
                tile_pred_sequeezed = tile_pred.squeeze() # Remove the channel dimension               

                pred_mask[y:y_end, x:x_end] = np.maximum(pred_mask[y:y_end, x:x_end], tile_pred_sequeezed) # Combine overlapping tiles by taking the maximum value

        return pred_mask

    def _resize_and_pad_mask(self, pred_mask: np.ndarray, bbox: tuple, full_size: tuple):
        """Resize the predicted mask and pad it to the original image size."""
        x_min, y_min, x_max, y_max = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
        cropped_height, cropped_width = y_max - y_min, x_max - x_min

        resized_mask = cv2.resize(pred_mask, (cropped_width, cropped_height))
        
        padded_mask = np.zeros(full_size[:2], dtype=np.uint8)

        padded_mask[y_min:y_max, x_min:x_max] = resized_mask

        return padded_mask

    def infer_single_image(self, image_path: str, _bbox: dict):
        """
        Performs segmentation mask prediction for a single image and saves the result.

        Args:
            image_path (str): Path to the input image.
        """
        log.info(f"Processing image: {image_path}")
        cropped_image, _, image_full_size = self._process_image(image_path, _bbox)
        cropped_image_shape = cropped_image.shape

        log.info(f"Size of cropped image: {cropped_image_shape}")

        if cropped_image_shape[0] < 4000 and cropped_image_shape[1] < 4000:
            log.info(f"Image size is small enough for direct processing.")
            pred_mask = self._predict_mask(cropped_image)
        else:
            log.info(f"Image size is too big for direct processing. Resizing and processing.")
            pred_mask = self._process_image_in_tiles(cropped_image) # Process in tiles

        padded_mask_for_visualization = self._resize_and_pad_mask(pred_mask, _bbox, image_full_size.shape[:2])

        log.info(f"inferencing completed for this image.")
        return padded_mask_for_visualization

    def save_image(self, img_path, _bbox):
        """
        Saves cropped images and masks based on the provided bounding box.

        Args:
            img_path (str): Path to the input image.
            _bbox (dict): Dictionary containing bounding box coordinates and class ID.
        """
        # Read the original image
        image = self._read_image(img_path)

        # Extract bounding box coordinates
        y_min, y_max, x_min, x_max, class_id = _bbox['y_min'], _bbox['y_max'], _bbox['x_min'], _bbox['x_max'], _bbox['class_id']

        # Generate mask from inference
        padded_mask = self.infer_single_image(img_path, _bbox)
        final_mask = np.where(padded_mask == 1, 255, 0)

        # Crop the mask to the bounding box and assign class ID
        final_mask = final_mask[y_min: y_max, x_min: x_max]
        final_mask[final_mask == 255] = class_id  

        # Convert the mask to a 3-channel image for visualization
        padded_mask_3d = np.repeat(padded_mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

        # Create the cutout image
        final_cutout_bgr = np.where(padded_mask_3d == 1, image, 0)
        final_cutout_rgb = cv2.cvtColor(final_cutout_bgr, cv2.COLOR_BGR2RGB)
        final_cutout_rgb = final_cutout_rgb[y_min: y_max, x_min: x_max]

        # Convert and crop the original image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_cropped = image_rgb[y_min: y_max, x_min: x_max]

        # Define save directory
        save_dir = Path(img_path).parent.parent / "cutouts"
        save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        # Generate filenames
        stem = Path(img_path).stem
        cropout_name = f"{stem}.jpg"
        final_mask_name = f"{stem}_mask.png"
        cutout_name = f"{stem}.png"

        # Save cropped image
        cv2.imwrite(str(save_dir / cropout_name), image_cropped.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 100])
        log.info(f"Cropped image saved as: {cropout_name}")

        # Save final mask
        cv2.imwrite(str(save_dir / final_mask_name), final_mask.astype(np.uint8))
        log.info(f"Final mask saved as: {final_mask_name}")

        # Save final cutout
        cv2.imwrite(str(save_dir / cutout_name), final_cutout_rgb.astype(np.uint8))
        log.info(f"Final cutout saved as: {cutout_name}")

    def process_image(self, input_paths):
        """
        Process an image and its corresponding JSON file.
        """
        log.info(f"Starting process to read image and process JSON file.")
        image_path, _ = input_paths
        json_path = Path(image_path).parent.parent / "cutouts" / f"{Path(image_path).stem}.json"
        log.info(f"Processing image: {image_path}")

        # Check if the metadata for the image exists
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
        
    def process_directory(self):
        """
        Processes all images in the test directory for segmentation inference.
        """
        batches = list(Path(self.temp_dir).iterdir())

        for batch in batches:
            img_dir = Path(batch / "developed_images")
            log.info(f"Processing images in directory: {img_dir}")
            images = sorted(list(img_dir.rglob("*.jpg")))
            for img_path in images:
                save_dir = Path(img_path).parent.parent / "cutouts"
                input_paths = (img_path, save_dir)
                try:
                    _, _bbox = self.process_image(input_paths)
                except Exception as e:
                    log.error(f"Error processing image: {img_path}")
                    log.error(e)
                    
                try:
                    self.save_image(img_path, _bbox)
                except Exception as e:
                    log.error(f"Error processing image: {img_path}")
                    log.error(e)

        log.info("Inference completed.")

def main(cfg: DictConfig):
    """
    Main function to initialize and start inference.

    Args:
        cfg (DictConfig): Configuration object.
    """
    log.info(f"Starting UNet segmentation inference.")
    trainer = UNetInference(cfg)
    trainer.process_directory()
