import json
import logging

from pathlib import Path
from ultralytics import YOLO
from omegaconf import DictConfig
from typing import Optional, Dict

# Configure logging
log = logging.getLogger(__name__)


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
        self.missing_detection_notes = [] # list to store missing detection notes
    
    def detect_weeds(self, image_path: Path) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Detects target weed in the given image.

        Parameters:
            image (np.ndarray): Image as a numpy array.

        Returns:
            dict: Detection results including bbox if detection is successful; otherwise, returns None.
        """
        log.info("Starting weed detection.")
        results = self.model(image_path)

        if not results or not results[0].boxes.xyxy.tolist():
            log.warning("No detection found.")
            self.missing_detection_notes.append("No detection found.")
            bbox = None
            det_pred_conf = None
        
        else:
            if len(results[0].boxes.xyxy.tolist()) > 1:
                log.warning("More than one detection found. Using the one with the highest confidence.")
                self.missing_detection_notes.append("More than one detection found. Using the one with the highest confidence.")
                # choose the detection with the highest confidence
                confidences = [x.conf for x in results[0].boxes]
                max_conf_idx = confidences.index(max(confidences))
                bbox = results[0].boxes.xyxy.tolist()[max_conf_idx]
                # confidence of the detection
                det_pred_conf = f"{results[0].boxes.conf[max_conf_idx].item():.3f}"
            else:
                # Extract the detection confidence score
                det_pred_conf = f"{results[0].boxes.conf.item():.3f}"
                # Extract the bounding box coordinates
                bbox = results[0].boxes.xyxy.tolist()[0]
            
            x_min, y_min, x_max, y_max = map(round, bbox)
            bbox_height = y_max - y_min
            bbox_width = x_max - x_min
            bbox = [x_min, y_min, bbox_width, bbox_height]
        return {
        "bbox": bbox,
        "det_pred_conf": det_pred_conf
        }

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
        self.output_dir = Path(cfg.paths.temp_dir)
        self.batch_id = cfg.batch_id
        self.weed_detector = WeedDetector(cfg.paths.yolo_weed_detection_model)

    def update_metadata(self, metadata_path: Path, metadata: dict) -> None:
        
        with open(metadata_path, "w") as file:
            json.dump(metadata, file, indent=4, default=str)
        
        log.debug(f"Metadata saved to {metadata_path.name}.")

    def load_metadata(self, metadata_path: Path) -> Dict:
        """
        This function loads the metadata from the given path.

        Parameters:
            metadata_path (Path): Path to the metadata file.

        Returns:
            dict: Loaded metadata.
        """
        with open(metadata_path, "r") as file:
            metadata = json.load(file)
        return metadata
    
    def update_notes(self, metadata: dict) -> dict:
        
        # Is there is already a note and missing detection notes
        if metadata["image_info"]["Note"] and self.weed_detector.missing_detection_notes:
            # If the missing detection note is already in the note
            if metadata["image_info"]["Note"] in self.weed_detector.missing_detection_notes:
                pass
            # If there is already a note and it is not in the missing detection notes
            else:
                metadata["image_info"]["Note"] = ". ".join(self.weed_detector.missing_detection_notes)
        
        # If there is no note and no missing detection notes
        elif metadata["image_info"]["Note"] == None and not self.weed_detector.missing_detection_notes:
            pass

        # If there is no note and there are missing detection notes
        elif metadata["image_info"]["Note"] == None and self.weed_detector.missing_detection_notes:
            metadata["image_info"]["Note"] = ". ".join(self.weed_detector.missing_detection_notes)

        # IF there is a note and no missing detection notes
        elif metadata["image_info"]["Note"] and not self.weed_detector.missing_detection_notes:
            pass
        
        else:
            raise ValueError("Error in updating notes.")

        return metadata


    def process_image_sequentially(self, image_path: Path) -> None:
        """
        This function processes the image by detecting weeds and saving the metadata.

        Parameters:
            image_path (Path): Path to the image file.

        Returns:
            None
        """
        batch_dir = image_path.parent.parent

        # Get metadata
        metadata_path = batch_dir / "cutouts" / f"{image_path.stem}_0.json"
        metadata = self.load_metadata(metadata_path)
        
        # Get "annotation" metadata
        detection_results = self.weed_detector.detect_weeds(image_path)
        metadata["annotation"]["bbox"] = detection_results.get("bbox", None)
        metadata["annotation"]["det_pred_conf"] = detection_results.get("det_pred_conf", None)

        # Update metadata Notess
        metadata = self.update_notes(metadata)
        
        # Save the image metadata
        self.update_metadata(
            metadata_path, 
            metadata)
    
    def process_images(self) -> None:
        """
        This function processes all the images in the directory.

        Returns:
            None
        """
        log.info(f"Processing images for {self.batch_id}.")
        image_dir = Path(self.output_dir / self.batch_id /"developed-images")
        image_metadata_dir = Path(self.output_dir) / self.batch_id /  'cutouts' # save metadata in the same batch as the image
        image_metadata_dir.mkdir(exist_ok=True)
        # Loop through the images in the batch
        image_paths = sorted(list(image_dir.glob("*.JPG")) + list(image_dir.glob("*.jpg")))
        # image_paths = [x for x in image_paths if x.stem == "ALA00118"]
        for image_path in image_paths:
            self.weed_detector.missing_detection_notes = []
            self.process_image_sequentially(image_path)
        log.info(f"Processed {len(image_paths)} images in {self.batch_id}.")
        

def main(cfg: DictConfig) -> None:
    """
    Main function to start the weed detection process.

    Parameters:
        cfg (DictConfig): Configuration object containing the paths to the data files.

    Returns:
        None
    """
    log.info(f"Starting {cfg.general.task}")
    processdetections = ProcessDetections(cfg)
    processdetections.process_images()
    log.info(f"{cfg.general.task} completed.")
