import cv2
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def rem_gray_bckgrnd(image_path, output_dir):
    try:
        # Read the image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Error reading image: {image_path}")
            return
        
        # Convert to RGB and HSV
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        # # Define the range of black and gray color in HSV
        # lower_gray_black = np.array([0, 10, 0])
        # upper_gray_black = np.array([180, 75, 255])

        # Define the range of black and gray color in HSV (%) 
        lower_gray_black = np.array([(0 * 180) / 360, (4 * 255) / 100, (0 * 255) / 100])
        upper_gray_black = np.array([(360 * 180) / 360, (30 * 255) / 100, (100 * 255) / 100])   

        # Create masks for the colors
        mask_gray_black = cv2.inRange(image_hsv, lower_gray_black, upper_gray_black)
        mask_not_black_gray = cv2.bitwise_not(mask_gray_black) # mask for not gray

        image_gray_rem_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask=mask_not_black_gray)
        image_gray_rem_rgb = cv2.cvtColor(image_gray_rem_hsv, cv2.COLOR_HSV2RGB)

        # Ostsu thresholding on top of masked gray_black
        image_gray = cv2.cvtColor(image_gray_rem_rgb, cv2.COLOR_BGR2GRAY)
        image_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
        _, otsu_mask = cv2.threshold(image_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        image_otsu_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask=otsu_mask)
        image_otsu_rgb = cv2.cvtColor(image_otsu_hsv, cv2.COLOR_HSV2RGB)

        # Generate output file path
        _, image_name = os.path.split(image_path)
        output_image_name = os.path.splitext(image_name)[0] + "_bckgrnd_rm.JPG"
        output_image_path = os.path.join(output_dir, output_image_name)

        #combined otsu and gray/black mask 
        combined_gray_black_otsu = cv2.bitwise_and(mask_not_black_gray, otsu_mask)
        image_gray_black_otsu_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask=combined_gray_black_otsu)
        image_gray_black_otsu_rgb = cv2.cvtColor(image_gray_black_otsu_hsv, cv2.COLOR_HSV2RGB)

        # Plot and save the result
        plt.figure(figsize=(20, 20))
        # plt.subplot(2, 2, 1)
        # plt.title("Original Image")
        # plt.axis("off")
        # plt.imshow(image_rgb)

        # plt.subplot(2, 2, 2)
        # plt.title("Gray and Black mask")
        plt.axis("off")
        plt.imshow(image_gray_rem_rgb)

        # plt.subplot(2, 2, 3)
        # plt.title("Combined otsu and gray and black mask ")
        # plt.axis("off")
        # plt.imshow(image_gray_black_otsu_rgb)

        # plt.subplot(2, 2, 4)
        # plt.title("Otsu threshold on top of gray and black masked image")
        # plt.axis("off")
        # plt.imshow(image_otsu_rgb)

        plt.savefig(output_image_path, dpi=100)
        plt.close()

        print(f"Processed and saved: {output_image_path}")

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

# Define directories
image_dir = "/home/nsingh27/Field-AnnotationPipeline/data/subset_images/test_gray_remove/input_dir"
output_dir = "/home/nsingh27/Field-AnnotationPipeline/data/subset_images/test_gray_remove/output_dir"
os.makedirs(output_dir, exist_ok=True)

# Process images in the input directory
valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
for fname in os.listdir(image_dir):
    if any(fname.endswith(ext) for ext in valid_extensions):
        image_path = os.path.join(image_dir, fname)
        rem_gray_bckgrnd(image_path, output_dir)
