import os
import logging
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path
import shutil
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
log = logging.getLogger(__name__)

class BatchDownloader:
    """
    A class to download image batches that have 0 cutouts and metadata.
    
    Attributes:
        cfg (DictConfig): The configuration object containing directory paths and settings.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the BatchDownloader with a configuration object.

        Args:
            cfg (DictConfig): Configuration object containing paths for CSV reports, batch storage,
                              and various settings like download limits and multithreading options.
        """
        self.cfg = cfg
        self.report_dir = Path(cfg.paths.reports)
        self.longterm_storage = Path(cfg.paths.longterm_storage)
        self.temp_storage = Path(cfg.paths.temp_dir)
        self.temp_storage.mkdir(parents=True, exist_ok=True)
        
    def find_most_recent_report(self) -> Optional[Path]:
        """
        Finds the most recent report file in the report directory based on the timestamp in the filename.

        Returns:
            Optional[Path]: The path of the most recent report file, or None if no reports are found.
        """
        # Get a list of all report files in the directory
        report_files = list(self.report_dir.glob("report_*.csv"))
        
        if not report_files:
            log.warning("No reports found in the report directory.")
            return None
        
        # Sort the report files based on the timestamp in their names
        most_recent_report = max(report_files, key=lambda f: f.stem.split('_')[1])
        relative_path = most_recent_report.relative_to(self.cfg.paths.workdir)
        log.info(f"Most recent report found: {relative_path}")
        
        return most_recent_report
    
    def load_report(self, report_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Loads the CSV report into a DataFrame.

        Args:
            report_path (Optional[Path]): Path to the specific report to load. If None, the most recent report is loaded.

        Returns:
            pd.DataFrame: A DataFrame containing the batch information from the CSV file.
        """
        # If no report_path is provided, find the most recent report
        if report_path is None:
            report_path = self.find_most_recent_report()
            if report_path is None:
                raise FileNotFoundError("No report available to load.")

        log.info(f"Loading report from {report_path.relative_to(self.cfg.paths.workdir)}")
        df = pd.read_csv(report_path)
        return df

    def filter_batches(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the DataFrame for batches that meet the following criteria:
        - 0 cutouts
        - 0 metadata
        - Equal number of raw and developed images
        - More than 0 raw images and more than 0 developed images
        Removes batches that are already fully downloaded and limits the result to download_limit.

        Args:
            df (pd.DataFrame): The DataFrame to filter.

        Returns:
            pd.DataFrame: A DataFrame containing the filtered and limited batches.
        """
        download_limit = self.cfg.download_batch.download_limit
        
        # Apply the initial filtering criteria
        filtered_df = df[
            (df['Cutouts'] == 0) & 
            (df['Metadata'] == 0) & 
            (df['Raw Images'] == df['Developed Images']) &
            (df['Raw Images'] > 0) &
            (df['Developed Images'] > 0)
        ]
        filtered_df = filtered_df.sort_values(by=['Batch'])
        
        # List to store indices of fully downloaded batches and those to be downloaded
        present_fully_processed_batches = []
        download_batches = []
        
        # Iterate over filtered_df and check for completeness
        for index, row in filtered_df.iterrows():
            batch_name = row["Batch"]
            target_batch_path = self.temp_storage / batch_name

            if target_batch_path.exists():
                log.info(f"Batch {batch_name} already exists. Checking for completeness.")
                if self.check_images_exist(target_batch_path, row['Developed Images']):
                    log.info(f"Batch {batch_name} is already fully downloaded.")
                    present_fully_processed_batches.append(index)
                    download_limit -= 1
                else:
                    log.warning(f"Batch {batch_name} is incomplete. Re-downloading the batch.")
                    shutil.rmtree(target_batch_path)  # Delete the incomplete folder to re-download it.
                    download_batches.append(index)
            else:
                download_batches.append(index)
        
        # Limit the DataFrame to the download limit
        filtered_df = filtered_df.loc[download_batches[:download_limit]]
        
        if filtered_df.empty:
            return None
        else:
            log.info(f"Filtered down to {len(filtered_df)} batches after removing fully downloaded ones and applying the download limit.")
            return filtered_df

    def check_images_exist(self, batch_path: Path, expected_image_count: int) -> bool:
        """
        Checks if all images are present in the batch folder and logs the number of images.

        Args:
            batch_path (Path): The path to the batch folder.
            expected_image_count (int): The number of images expected to be in the batch folder.

        Returns:
            bool: True if all images are present, False otherwise.
        """
        developed_batch_path = batch_path / "developed_images"
        image_files = list(developed_batch_path.glob("*.jpg"))  # Adjust the image extension as needed
        num_images = len(image_files)
        log.info(f"Batch {developed_batch_path.parent.name}/{developed_batch_path.name} contains {num_images}/{expected_image_count} images.")
        return num_images == expected_image_count
    
    def download_image(self, src: Path, dest: Path) -> None:
        """
        Downloads a single image from source to destination.

        Args:
            src (Path): The source path of the image.
            dest (Path): The destination path where the image will be copied.
        """
        try:
            shutil.copy2(src, dest)
        except Exception as e:
            log.exception(f"Failed to download image {src.name}: {e}")

    def download_batch(self, batch_name: str, expected_image_count: int) -> None:
        """
        Downloads the batch by copying images to temp storage if not already downloaded or incomplete,
        while skipping the 'raws' directory and using multithreading per image.

        Args:
            batch_name (str): The name of the batch folder to be downloaded.
            expected_image_count (int): The expected number of images in the batch folder.
        """
        source_batch_path = self.longterm_storage / batch_name
        target_batch_path = self.temp_storage / batch_name

        if source_batch_path.exists():
            try:
                log.info(f"Downloading batch {batch_name} to {target_batch_path.parent.parent.name}/{target_batch_path.parent.name}/{target_batch_path.name}, skipping 'raws' directory.")
                target_batch_path.mkdir(parents=True, exist_ok=True)
                
                with ThreadPoolExecutor() as executor:
                    futures = []
                    
                    for item in source_batch_path.iterdir():
                        if item.name == "raws" and item.is_dir():
                            log.info(f"Skipping 'raws' directory in batch {batch_name}.")
                            continue

                        if item.is_dir():
                            dest_dir = target_batch_path / item.name
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            for sub_item in item.iterdir():
                                futures.append(executor.submit(self.download_image, sub_item, dest_dir / sub_item.name))
                        else:
                            futures.append(executor.submit(self.download_image, item, target_batch_path / item.name))

                    for future in as_completed(futures):
                        future.result()
            except Exception as e:
                log.exception(f"Failed to download batch {batch_name}: {e}")
        else:
            log.error(f"Batch {batch_name} does not exist in long-term storage.")

    def process_batches(self) -> None:
        """
        Processes and downloads batches that meet the filtering criteria, with a limit on the number of downloads.
        Utilizes multithreading to download batches concurrently if enabled in the configuration.
        """
        df = self.load_report()
        filtered_batches = self.filter_batches(df)

        if filtered_batches is None:
            log.info(f"Number of downloaded batches exceeds the download limit ({self.cfg.download_batch.download_limit}). Stopping the script.")
            return 
        
        # Limit the number of batches to download
        download_limit = self.cfg.download_batch.download_limit
        log.info(f"Download limit set to {download_limit} batches.")
        
        # Check if multithreading is enabled in the config
        use_multithreading = self.cfg.download_batch.use_multithreading
        
        if use_multithreading:
            log.info("Multithreading is enabled. Downloading batches concurrently.")
            max_workers = int(len(os.sched_getaffinity(0)) / 5)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {
                    executor.submit(self.download_batch, row['Batch'], row['Raw Images']): row['Batch']
                    for _, row in filtered_batches.iterrows()
                }
                
                for future in as_completed(future_to_batch):
                    batch_name = future_to_batch[future]
                    try:
                        future.result()
                        log.info(f"Batch {batch_name} downloaded successfully.")
                    except Exception as e:
                        log.error(f"Batch {batch_name} generated an exception: {e}")
        else:
            log.info("Multithreading is disabled. Downloading batches sequentially.")
            for _, row in filtered_batches.iterrows():
                batch_name = row['Batch']
                expected_image_count = row['Raw Images']
                self.download_batch(batch_name, expected_image_count)


def main(cfg: DictConfig) -> None:
    log.info(f"Starting batch download process.")
    
    # Initialize the BatchDownloader with the provided configuration
    batch_downloader = BatchDownloader(cfg)
    
    # Process and download the filtered batches
    batch_downloader.process_batches()
    
    log.info(f"Batch download process completed.")
