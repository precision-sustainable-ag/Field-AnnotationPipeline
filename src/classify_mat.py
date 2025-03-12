import json
import logging
import pandas as pd

from pathlib import Path
from ultralytics import YOLO
from omegaconf import DictConfig
from typing import Optional, Dict

# Configure logging
log = logging.getLogger(__name__)


class MatClassifier:
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
    
    def classify_image(self, image: Path) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Classify  mat in the given image.

        Parameters:
            image (np.ndarray): Image as a numpy array.

        Returns:
            dict: Detection results including bbox if detection is successful; otherwise, returns None.
        """
        log.debug("Starting mat classification.")
        # Predict classes and probabilities
        results = self.model(image)  # predict on an image
        assert len(results) == 1, "Only one image should be processed at a time."

        for result in results:
            r = result.probs
            top1 =  r.top1
            top1conf =  float(r.top1conf)
            hasmat = True if top1 == 0 else False
            result = {
                "HasMatPred": hasmat, 
                "HasMatPredConf": top1conf
                }
            return result

        # return class_results

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
        self.weed_classifier = MatClassifier(cfg.paths.yolo_mat_classifier)

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

    def process_image_sequentially(self, image_path: Path) -> None:
        """
        This function processes the image by detecting weeds and saving the metadata.

        Parameters:
            image_path (Path): Path to the image file.

        Returns:
            None
        """
        image_name = image_path.name
        batch_dir = image_path.parent.parent
        # Get metadata
        metadata_path = batch_dir / "cutouts" / f"{image_path.stem}_0.json"
        metadata = self.load_metadata(metadata_path)

        # Get "annotation" metadata
        detection_results = self.weed_classifier.classify_image(image_path)
        metadata["annotation"]["HasMatPred"] = detection_results["HasMatPred"]
        metadata["annotation"]["HasMatPredConf"] = detection_results["HasMatPredConf"]
        
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
        for image_path in image_paths:
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
