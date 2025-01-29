import os
import cv2
import shutil
import numpy as np
import pandas as pd
from pathlib import Path


class ImageCutoutReviewer:
    """
    A class to manage the process of reviewing image cutouts, collecting user ratings, and storing selected images and masks 
    for further segmentation training.

    The process involves:
    - Filtering metadata to include only relevant images.
    - Displaying images (cutout and cropout) side by side for user review.
    - Saving images and corresponding masks that are rated as 'good' cutouts to the YOLO segmentation dataset folder.
    """

    def __init__(self, image_folder, metadata_csv, yolo_segmentation_folder, ratings_folder):
        """
        Initializes the ImageCutoutReviewer class with paths to the necessary folders and files.

        Parameters:
        - image_folder (str or Path): Path to the folder containing images.
        - metadata_csv (str or Path): Path to the CSV file containing image metadata.
        - yolo_segmentation_folder (str or Path): Path to the folder where YOLO segmentation images and masks will be stored.
        - ratings_folder (str or Path): Path to the folder where the ratings CSV will be stored.
        """
        self.image_folder = Path(image_folder)
        self.metadata_csv = Path(metadata_csv)
        self.yolo_segmentation_folder = Path(yolo_segmentation_folder)
        self.ratings_folder = Path(ratings_folder)
        self.ratings_folder.mkdir(exist_ok=True, parents=True)  # Create the ratings folder if it does not exist

        # Load metadata CSV file
        self.df = pd.read_csv(self.metadata_csv, low_memory=False)
        # Get the batch folder name from the image folder path
        self.batch_name = self.image_folder.parts[-2]
        # Set the output CSV file for storing cutout ratings
        self.rating_csv = self.ratings_folder / f"cutout_ratings_{self.batch_name}.csv"

    def df_filter(self):
        """
        Filters the metadata DataFrame to include only rows corresponding to images present in the image folder.

        This function scans the image folder for image files (with extensions .jpg or .png) and filters the metadata
        to include only those images, ensuring the user is presented only with relevant image data.

        Returns:
        - pd.DataFrame: A filtered DataFrame containing metadata of images present in the image folder.
        """
        print(f"Filtering DataFrame for images present in {self.image_folder}")
        # Get list of image filenames from the folder (only .jpg and .png files)
        images = [Path(img).name for img in os.listdir(self.image_folder) if img.lower().endswith((".jpg", ".png"))]
        # Filter the DataFrame for images found in the folder
        df_filtered = self.df[self.df["Name"].isin(images)]
        return df_filtered

    def image_viewer(self, df_filtered):
        """
        Displays images (cutout and cropout) side by side for user rating and saves the ratings to a CSV file.

        This function shows the cutout images and their cropout versions with a mask overlay, allowing the user to rate
        the quality of the cutout. The ratings (good: 1, bad: 0) are saved in a CSV file.

        Parameters:
        - df_filtered (pd.DataFrame): Filtered DataFrame containing image names and metadata.

        Returns:
        - None: Ratings are saved to the specified CSV file.
        """
        print("Starting image viewer for cutout quality rating.")
        results_list = []
        image_number = 0

        for _, row in df_filtered.iterrows():
            image_number += 1
            print(f"Image {image_number} of {len(df_filtered)}")
            image_name = row["Name"]
            image_path_cutout = os.path.join(self.image_folder, image_name)
            image_path_cropout = image_path_cutout.replace(".jpg", "_cropout.jpg").replace(".JPG", "_cropout.JPG")

            # Load the cutout and cropout images
            cutout_image = cv2.imread(image_path_cutout)
            cropout_image = cv2.imread(image_path_cropout)

            if cutout_image is None or cropout_image is None:
                print(f"Could not load images for {image_name}. Skipping.")
                continue

            # Resize images for easier viewing
            cutout_image_resized = cv2.resize(cutout_image, (cutout_image.shape[1] // 10, cutout_image.shape[0] // 10))
            cropout_image_resized = cv2.resize(cropout_image, (cropout_image.shape[1] // 10, cropout_image.shape[0] // 10))

            # Create a grayscale version of the cutout image and generate a binary mask
            cutout_gray = cv2.cvtColor(cutout_image_resized, cv2.COLOR_BGR2GRAY)
            cutout_mask = np.where(cutout_gray == 0, 0, 255).astype(np.uint8)

            # Apply a random color to the mask
            # color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
            # Purple color
            color = (128, 0, 128)
            colored_mask = np.zeros_like(cutout_image_resized)
            for c in range(3):
                colored_mask[:, :, c] = np.where(cutout_mask == 255, color[c], 0)

            # Overlay the mask on the cropout image
            overlay_mask = cv2.addWeighted(cropout_image_resized, 1, colored_mask, 0.8, 0)

            # Show the image and collect user rating
            valid_input = False
            while not valid_input:
                cv2.imshow("Rate the cutout quality (0: bad, 1: good)", overlay_mask)
                key = cv2.waitKey(0) & 0xFF

                if key in [ord('0'), ord('1')]:
                    rating = key - ord('0')
                    results_list.append({
                        "Name": image_name,
                        "Species": row["Species"],
                        "Cutout_quality": rating
                    })
                    valid_input = True
                elif key == 27:  # ESC key to skip
                    print(f"Skipping image {image_name}.")
                    valid_input = True

            cv2.destroyAllWindows()

        # Save the ratings to a CSV file
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(self.rating_csv, index=False)
        print(f"Ratings saved to {self.rating_csv}")

    def save_images_masks_for_good_cutouts(self):
        """
        Saves images and corresponding masks of good cutouts (rated 1) to the YOLO segmentation folder.

        After the user rates the cutouts, this function filters for images rated as 'good' and copies both the
        image and its corresponding mask to the designated YOLO segmentation dataset folders.

        Returns:
        - None: Files are copied to the YOLO segmentation folders.
        """
        print(f"Saving images and masks for good cutouts to {self.yolo_segmentation_folder}")

        # Load the ratings CSV file
        df = pd.read_csv(self.rating_csv)
            
        # Filter for images with a rating of 1 (good cutouts)
        df_good_cutouts = df[df['Cutout_quality'] == 1]
        # Create a list of good cutout names without file extensions
        good_cutout_list = df_good_cutouts['Name'].str.replace('.JPG', '').values.tolist()

        # Define the paths to save masks and images in the YOLO segmentation folder
        good_masks_folder = self.yolo_segmentation_folder / 'masks'
        chosen_images_folder = self.yolo_segmentation_folder / 'images'

        # Ensure the directories exist
        good_masks_folder.mkdir(exist_ok=True, parents=True)
        chosen_images_folder.mkdir(exist_ok=True, parents=True)

        # Get the batch folder and full-sized masks folder
        batch_folder = self.image_folder.parent
        full_masks_folder = batch_folder / "fullsized_masks"

        if full_masks_folder.exists():
            for full_mask in full_masks_folder.iterdir():
                full_mask_name = full_mask.stem
                if full_mask_name in good_cutout_list:
                    # Copy mask and image to their respective YOLO segmentation folders
                    shutil.copy(full_mask, good_masks_folder / full_mask.name)
                    shutil.copy(batch_folder / 'developed-images' / f'{full_mask_name}.jpg', chosen_images_folder / f'{full_mask_name}.jpg')
                else:
                    print(f'No good cutout found for {full_mask_name}')
        else:
            print(f'No fullsized masks folder found.')

    def run(self):
        """
        Executes the entire cutout review process:
        1. Filters the metadata DataFrame to include only images present in the folder.
        2. Launches the image viewer for user ratings.
        3. Saves images and masks for good cutouts to the YOLO segmentation folder.
        """
        df_filtered = self.df_filter()
        print(f'Number of images to label: {len(df_filtered)}')
        self.image_viewer(df_filtered)
        self.save_images_masks_for_good_cutouts()


# Define paths for image folder, YOLO segmentation folder, and metadata CSV
image_folder = "/home/nsingh27/Field-AnnotationPipeline/data/temp/NC_2022-07-27/cutouts"
metadata_csv = "/home/nsingh27/Field-AnnotationPipeline/data/persistent_tables/merged_blobs_tables_metadata_permanent.csv"
yolo_segmentation_folder = "/home/nsingh27/Field-AnnotationPipeline/data/yolo_segmentation_dataset"
ratings_folder = '/home/nsingh27/Field-AnnotationPipeline/cutout_ratings'

# Instantiate the class and run the process
reviewer = ImageCutoutReviewer(image_folder, metadata_csv, yolo_segmentation_folder, ratings_folder)
reviewer.run()
