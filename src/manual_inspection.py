import cv2
import hydra
import getpass
import logging
import datetime
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
import time

# Configure logging
log = logging.getLogger(__name__)

GITHUB_REPO_URL = "https://github.com/precision-sustainable-ag/Field-AnnotationPipeline/issues"

LABEL_OPTIONS = {
                "1": "Good Mask",
                "2": "Bad Mask",
                "3": "Incorrect Species",
                "0": "Other",
                "q": "Quit"
            }

class ManualInspection:
    """
    Class to manually inspect images and label them based on quality.
    """
    def __init__(self, cfg: DictConfig):
        self.batch_id = cfg.batch_id
        self.longterm_inspection_dir = Path(cfg.paths.longterm_storage) / "field-batches"/ self.batch_id/ "inspection"
        self.csv_file = self.longterm_inspection_dir / f"{self.batch_id}_preprocessing_inspection_results.csv"
        self.images = self._load_images()
        self.results = self._load_existing_results()
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.user = getpass.getuser()

    def _load_images(self):
        """Load images and return a sorted list of unlabeled ones."""
        all_images = sorted(self.longterm_inspection_dir.glob("*.jpg"))
        if not all_images:
            log.warning(f"No images found in {self.longterm_inspection_dir}")
            return []

        return [img for img in all_images if img.stem not in self._get_labeled_images()]

    def _get_labeled_images(self):
        """Retrieve a set of already labeled images from the CSV file."""
        if self.csv_file.exists():
            df_existing = pd.read_csv(self.csv_file)
            return set(df_existing['ImageID'].tolist())
        return set()

    def _load_existing_results(self):
        """Load existing CSV results or return an empty list."""
        if self.csv_file.exists():
            log.info(f"Loading existing results from {self.csv_file}")
            return pd.read_csv(self.csv_file).values.tolist()
        return []

    def _display_instructions(self):
        """Prints instructions for user input."""
        print("\n--- Segmentation Quality Assessment ---")
        print_labels = {
            "1": "Good Mask",
            "2": "Bad Mask",
            "3": "Incorrect Species",
            "0": "Other",
            "q": "Quit"
        }
        
        for key, label in print_labels.items():
            if key == "0":
                print(f"{key}️ (zero) - {label}")
            else:
                print(f"{key}️ - {label}")
        print("\n🔄 Please wait while the X11 or X410 forwarding initializes. This may take a few seconds...\n")

    def _display_image(self, img_path):
        """Loads and displays an image, returns False if loading fails."""
        image = cv2.imread(str(img_path))
        if image is None:
            log.error(f"⚠️ Error loading image: {img_path}")
            return False
        
        cv2.imshow("Inspection Viewer", image)
        return True

    def _get_user_input(self):
        """Captures user input for labeling images."""
        while True:
            key = cv2.waitKey(0) & 0xFF
            key_char = chr(key)
            if key_char in LABEL_OPTIONS:
                return LABEL_OPTIONS[key_char]
            print("⚠️ Invalid choice. Please press a number between 0-3 or 'q' to quit.")

    def _save_results(self):
        """Save the labeling results to a CSV file."""
        df = pd.DataFrame(self.results, columns=['BatchID', 'ImageID', 'Selection', 'Timestamp', 'User', 'LTSLocation'])
        df.to_csv(self.csv_file, index=False)

    def _review_flagged_images(self):
        """Checks and offers to display flagged images for issue reporting."""
        df_final = pd.read_csv(self.csv_file)
        flagged_images = df_final[df_final["Selection"] != "Good Mask"]

        if flagged_images.empty:
            return self.csv_file

        print("\n⚠️ Some images have issues.")
        print(f"📌 Please report in our GitHub repository: {GITHUB_REPO_URL}")
        print("Mention the flagged images and describe the issues.")
        if input("Would you like to review the flagged images for screenshots? (y/n): ").strip().lower() == 'y':
            print("Waiting to allow the image viewer to load...")
            time.sleep(10)
            self._display_flagged_images(flagged_images)

        print("\n📌 After taking screenshots, submit an issue on GitHub:")
        print(f"🔗 {GITHUB_REPO_URL}\n")
        print(f"Title the issue: {self.batch_id} segmentation quality inspection: {len(flagged_images)} flagged images\n")
        return self.csv_file

    def _display_flagged_images(self, flagged_images):
        """Displays flagged images for screenshot capture."""
        for _, row in flagged_images.iterrows():
            img_path = self.longterm_inspection_dir / f"{row['ImageID']}.jpg"
            if not img_path.exists():
                img_path = self.longterm_inspection_dir / f"{row['ImageID']}.JPG"

            if img_path.exists():
                image = cv2.imread(str(img_path))
                cv2.imshow("Flagged Image", image)
                print(f"📸 Take a screenshot for: {row['ImageID']} ({row['Selection']})\n")
                print("Press any key to continue to the next image.")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):  # Allow early exit
                    break
            else:
                print(f"⚠️ Could not find image: {row['ImageID']}")

        cv2.destroyAllWindows()

    def review_images(self):
        """Iterate over images and allow the user to label them."""
        if not self.images:
            log.info("✅ All images have been labeled. Exiting.")
            return None

        self._display_instructions()
        cv2.namedWindow("Inspection Viewer")

        index = 0
        while index < len(self.images):
            img_path = self.images[index]
            if not self._display_image(img_path):
                index += 1
                continue

            label = self._get_user_input()
            if label == "Quit":
                print("\n❌ Exiting image review.")
                cv2.destroyAllWindows()
                return self.csv_file  # Save progress and exit

            self.results.append([self.batch_id, img_path.stem, label, self.timestamp, self.user, self.longterm_inspection_dir])
            self._save_results()
            index += 1

        cv2.destroyAllWindows()
        log.info("✅ Segmentation quality inspection completed.")
        return self._review_flagged_images()

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to start the manual inspection process.
    """
    # Initialize the ManualInspection class
    log.info("Starting the manual inspection process.")
    manual_inspection = ManualInspection(cfg)
    src_csv_file = manual_inspection.review_images()
    log.info(f"Inspection results saved to {src_csv_file}")
    log.info("Image inspection completed.")
