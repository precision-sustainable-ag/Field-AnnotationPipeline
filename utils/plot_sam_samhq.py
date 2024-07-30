import matplotlib.pyplot as plt
import os
import cv2

# Directories
dir1 = r"data/images_testing/subset/output/output_sam_hq_on_og_image_mask_more_than_0.5"
dir2 = r"data/images_testing/subset/output/output_sam_hq_on_padded_cropout_mask_less_than_0.5"
dir3 = r"data/images_testing/subset/output/output_sam_hq_on_padded_cropout_mask_more_than_0.5_prefered"
dir4 = r"data/images_testing/subset/output/output_sam_on_og_image_mask_more_than_0.5"
dir5 = r"data/images_testing/subset/output/output_sam_on_padded_cropout_mask_less_than_0.5"
dir6 = r"data/images_testing/subset/output/output_sam_on_padded_cropout_mask_more_than_0.5_prefered"

# List of directories
dirs = [dir1, dir2, dir3, dir4, dir5, dir6]

# Collect filenames from the first directory
filenames = os.listdir(dir1)

for sam_image_name in filenames:
    if sam_image_name.endswith("cutout.png"):
        # Initialize a list to hold the images
        images = []
        titles = [
            "sam_hq_on_og_image_mask_more_than_0.5",
            "sam_hq_on_padded_cropout_mask_less_than_0.5",
            "sam_hq_on_padded_cropout_mask_more_than_0.5_correct",
            "sam_on_og_image_mask_more_than_0.5",
            "sam_on_padded_cropout_mask_less_than_0.5",
            "sam_on_padded_cropout_mask_more_than_0.5_correct"
        ]

        for dir_path in dirs:
            img_path = os.path.join(dir_path, sam_image_name)
            if os.path.exists(img_path):
                cutout = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                images.append(cutout)
            else:
                images.append(None)

        # Create the plot
        plt.figure(figsize=(45, 20))

        for idx, (img, title) in enumerate(zip(images, titles)):
            if img is not None:
                plt.subplot(2, 3, idx + 1)
                plt.imshow(img)
                plt.title(title, fontsize=30)
                plt.gca().axis("off")

        # Save the figure
        save_fig_path = os.path.join(r"data/images_testing/subset/output/compare_sam_different", sam_image_name)
        plt.savefig(save_fig_path, dpi=100)
        plt.close()




