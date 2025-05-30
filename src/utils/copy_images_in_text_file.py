import shutil
from pathlib import Path

def copy_images_from_list(input_dir: Path, output_dir: Path, input_txt_file: Path):
    """
    Copies images listed in a text file from a source directory (recursively) to a destination directory.

    Args:
        input_dir (Path): The root directory to search for images.
        output_dir (Path): The directory where found images will be copied.
        input_txt_file (Path): Path to a text file containing image filenames (one per line).

    The function reads image filenames from the text file, searches for each image in input_dir recursively,
    and copies found images to output_dir. If an image is not found, it prints a warning.
    """
    # Read image names from the txt file
    with open(input_txt_file, 'r') as f:
        image_names = [line.strip() for line in f if line.strip()]
    for image_name in image_names:
        found = False
        for image_path in input_dir.rglob(image_name):
            if image_path.is_file():
                shutil.copy(image_path, output_dir / image_name)
                found = True
                break
        if not found:
            print(f"Image not found: {image_name}")

if __name__ == "__main__":
    input_directory = Path("/path/to/input/directory")
    output_directory = Path("/path/to/output/directory")
    input_text_file = Path("/path/to/input.txt")
    copy_images_from_list(input_directory, output_directory, input_text_file)
