import logging
from pathlib import Path
from omegaconf import DictConfig
import pandas as pd
import datetime
from typing import List, Dict

# Configure logging
log = logging.getLogger(__name__)

class BatchChecker:
    """
    A class to check the status of image batches in long-term storage.

    Attributes:
        cfg (DictConfig): The configuration object containing directory paths.
        longterm_storage (Path): Path to the long-term storage directory.
        report_dir (Path): Path to the directory where reports will be saved.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.longterm_storage = Path(self.cfg.data.longterm_storage)

        # Ensure the report directory exists
        self.report_dir = Path(cfg.reports)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def find_batches(self) -> List[str]:
        """
        Finds all batch directories within the long-term storage.

        Returns:
            List[str]: A list of batch folder names.
        """
        batches = []
        longterm_storage_path = Path(self.longterm_storage)
        
        # Iterate through directories in the long-term storage path
        for batch_folder in longterm_storage_path.iterdir():
            if batch_folder.is_dir():
                batches.append(batch_folder.name)
        return batches

    def check_folders(self) -> List[Dict[str, int]]:
        """
        Checks the content of each batch folder for raw images, developed images, metadata, and cutouts.

        Returns:
            List[Dict[str, int]]: A list of dictionaries, each containing the count of files in the batch's subdirectories.
        """
        report = []
        batches = self.find_batches()
        longterm_storage_path = Path(self.longterm_storage)  # Define long-term storage path

        # Iterate through each batch folder and gather information about its contents
        for batch_folder in batches:
            batch_path = longterm_storage_path / batch_folder
            developed_images_path = batch_path / "developed-images"
            raw_images_path = batch_path / "raws"
            metadata_path = batch_path / "metadata"
            cutouts_path = batch_path / "cutouts"

            # Create a report for the current batch
            batch_report: Dict[str, int] = {
                "Batch": batch_folder,
                "Raw Images": len(list(raw_images_path.rglob("*.ARW"))),
                "Developed Images": len(list(developed_images_path.glob("*.jpg"))),
                "Metadata": len(list(metadata_path.glob("*.json"))),
                "Cutouts": len(list(cutouts_path.glob("*.png"))),
            }
            report.append(batch_report)
    
        return report

    def save_report_as_csv(self, report: List[Dict[str, int]]) -> None:
        """
        Saves the report as a CSV file in the report directory.

        Args:
            report (List[Dict[str, int]]): The report data to be saved.
        """
        # Generate a timestamped filename for the report
        now = datetime.datetime.now()
        file_path_with_datetime = Path(self.report_dir, f"report_{now.strftime('%Y%m%d_%H%M%S')}.csv")
        
        # Convert the report data to a DataFrame and save it as a CSV file
        df = pd.DataFrame(report)
        df.to_csv(file_path_with_datetime, index=False)


def main(cfg: DictConfig) -> None:
    """Main function to start the batch checking process."""
    log.info(f"Starting {cfg.general.task}")
    
    # Initialize the BatchChecker with the provided configuration
    folder_checker = BatchChecker(cfg)
    
    # Check the contents of batch folders and generate a report
    report = folder_checker.check_folders()
    
    # Save the report to a CSV file
    folder_checker.save_report_as_csv(report)
    
    log.info(f"{cfg.general.task} completed.")
