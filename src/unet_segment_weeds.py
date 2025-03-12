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
        log.debug(f"Initializing UNetInference at {datetime.now()}")
        
        # Set up directories
        self.temp_dir = Path(cfg.paths.temp_dir)
        self.batch_id = cfg.batch_id
        self.batch_dir = self.temp_dir / self.batch_id
        self.trained_model_path = cfg.paths.unet_segmentation_model
        
        # Load UNet model
        self.seg_model = UNet(in_channels=3, num_classes=1).to(device)
        self.seg_model.load_state_dict(torch.load(self.trained_model_path, map_location=device, weights_only=True))
        self.seg_model.eval()

        # Define save directory
        self.cutout_dir = self.batch_dir / "cutouts"
        self.developed_dir = self.batch_dir / "developed-images"

        # Define image transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _read_image(self, image_path: Path) -> np.ndarray:
        log.debug(f"Reading image and converting to RGB: {image_path}")
        return cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

    def read_metadata(self, json_path):
        """
        Read the metadata from the JSON file.
        """
        if not json_path.exists():
            log.error(f"JSON file not found: {json_path}")
            return None

        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return data

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
        x_min, x_max = bbox["x_min"], bbox["x_max"]
        y_min, y_max = bbox["y_min"], bbox["y_max"]
        cropped_width = x_max - x_min
        cropped_height = y_max - y_min
        
        resized_mask = cv2.resize(pred_mask, (cropped_width, cropped_height))
        padded_mask = np.zeros(full_size[:2], dtype=np.uint8)
        padded_mask[y_min:y_max, x_min:x_max] = resized_mask

        return padded_mask

    def get_bbox_minmax(self, bbox: dict):
        """Crop the image based on the provided bounding box."""
        bbox_dict = {}

        y_min, y_max = bbox[1], bbox[1] + bbox[3]
        x_min, x_max = bbox[0], bbox[0] + bbox[2]
        bbox_dict["y_min"] = y_min
        bbox_dict["y_max"] = y_max
        bbox_dict["x_min"] = x_min
        bbox_dict["x_max"] = x_max
        return bbox_dict
    
    def pred_mask(self, cropped_image: np.ndarray):
        if cropped_image.shape[0] < 4000 and cropped_image.shape[1] < 4000:
            pred_mask = self._predict_mask(cropped_image)
        else:
            pred_mask = self._process_image_in_tiles(cropped_image) # Process in tiles
        return pred_mask

    def save_image(
            self, 
            img_path: str, 
            image_cropped: np.ndarray, 
            final_mask: np.ndarray, 
            final_cutout_rgb: np.ndarray
            ):
        
        # Generate filenames
        stem = Path(img_path).stem
        cropout_name = f"{stem}_0.jpg"
        final_mask_name = f"{stem}_0_mask.png"
        cutout_name = f"{stem}_0.png"

        # Save cropped image
        cv2.imwrite(str(self.cutout_dir / cropout_name), image_cropped.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 100])
        log.debug(f"Cropped image saved: {str(self.cutout_dir / cropout_name)}")

        # Save final mask
        cv2.imwrite(str(self.cutout_dir / final_mask_name), final_mask.astype(np.uint8))
        log.debug(f"Final mask saved: {str(self.cutout_dir / final_mask_name)}")

        # Save final cutout
        cv2.imwrite(str(self.cutout_dir / cutout_name), final_cutout_rgb.astype(np.uint8))
        log.debug(f"Final cutout saved: {str(self.cutout_dir / cutout_name)}")

    
    
    def process_image(self, input_paths):
        """
        Process an image and its corresponding JSON file.
        """
        image_path, json_path = input_paths

        # Check if the metadata for the image exists
        metadata = self.read_metadata(json_path)
        
        bbox = metadata["annotation"]["bbox"]
        if bbox is None:
            log.error(f"No bounding box found for {image_path}. Skipping.")
            return 
        
        categroy = metadata["category"]
        if categroy is None:
            log.error(f"No Category found for {image_path}. Skipping.")
            return
        
        class_id = categroy["class_id"]
        
        # Add min/max coordinates to the bounding box
        bx = self.get_bbox_minmax(bbox)
        
        # Read the original image
        image = self._read_image(image_path)
        image_cropped = image[bx["y_min"]:bx["y_max"], bx["x_min"]:bx["x_max"]]

        # Perform segmentation inference
        pred_mask = self.pred_mask(image_cropped)

        # Resize and pad the mask to the original image size
        padded_mask = self._resize_and_pad_mask(pred_mask, bx, image.shape[:2])

        # Crop the mask to the bounding box and assign class ID
        padded_cropped_mask = padded_mask[bx["y_min"]:bx["y_max"], bx["x_min"]:bx["x_max"]]
        final_mask = np.where(padded_cropped_mask == 1, class_id, 0)

        # Convert the mask to a 3-channel image for visualization
        padded_mask_3d = np.repeat(padded_mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

        # Create the cutout image
        cropped_padded_mask = padded_mask_3d[bx["y_min"]:bx["y_max"], bx["x_min"]:bx["x_max"]]
        # Convert image back to bgr
        image_cropped_bgr = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)
        final_cutout_rgb = np.where(cropped_padded_mask == 1, image_cropped_bgr, 0)

        self.save_image(image_path, image_cropped_bgr, final_mask, final_cutout_rgb)

        log.info(f"Image {image_path} processed.")
           
        
    def process_directory(self):
        """
        Processes all images in the test directory for segmentation inference.
        """
        images = sorted(list(self.developed_dir.glob("*.jpg")))
        log.info(f"Processing {len(images)} images in batch {self.batch_id}.")
        
        for img_path in images:
            json_path = self.cutout_dir / f"{img_path.stem}_0.json"
            input_paths = (img_path, json_path)
            self.process_image(input_paths)

def main(cfg: DictConfig):
    """
    Main function to initialize and start inference.
    """
    log.info(f"Starting UNet segmentation inference.")
    trainer = UNetInference(cfg)
    trainer.process_directory()
    log.info(f"UNet segmentation inference complete.")
