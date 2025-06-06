import shutil
from pathlib import Path

def copy_developed_images_from_longterm_storage(destination_dir, txt_files_with_names_images_needed):
    """
    Copies specific developed images from long-term storage to a destination directory.

    This function reads a text file containing the names of images to be copied. It then searches through
    batch directories in the long-term storage location, and for each batch, looks for a "developed-images"
    subdirectory. If the subdirectory exists, it copies any images whose names match those listed in the
    provided text file to the specified destination directory.

    Args:
        destination_dir (Path): The directory where the selected images will be copied.
        txt_files_with_names_images_needed (Path): Path to a text file containing image filenames (one per line)
            that need to be copied from long-term storage.

    Returns:
        None
    """
    longterm_storage_batches_dir = Path("/mnt/research-projects/r/raatwell/longterm_images3/field-batches")
    # Read the text file to get the list of image names
    images_names_to_copy = txt_files_with_names_images_needed.read_text().splitlines()

    # Iterate through each batch directory in the longterm storage
    for batch_dir in longterm_storage_batches_dir.iterdir():
        developed_images_dir = batch_dir / "developed-images"
        if developed_images_dir.exists():
            # Iterate through each image in the developed-images directory
            for image_file in developed_images_dir.iterdir():
                if image_file.name in images_names_to_copy:
                    # Copy the image to the destination directory
                    shutil.copy(image_file, destination_dir / image_file.name)
                    print(f"Copied {image_file.name} to {destination_dir}")

# Execute the function with specified paths
destination_dir = Path("/home/nsingh27/Field-AnnotationPipeline/data/temp/non_green_stem/developed-images")
txt_files_with_names_images_needed = Path("/home/nsingh27/Field-SegmentationTraining/data/IMP_non_green_stem_issue/20250606_non_green_stem_imporovement.txt")

copy_developed_images_from_longterm_storage(destination_dir, txt_files_with_names_images_needed)