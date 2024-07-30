import json
from PIL import Image
from PIL.ExifTags import TAGS
import os

# Function to extract EXIF data
def get_exif_data(image_path):
    image = Image.open(image_path)
    exif_data = {}
    if hasattr(image, '_getexif'):
        exif_info = image._getexif()
        if exif_info is not None:
            for tag, value in exif_info.items():
                tag_name = TAGS.get(tag, tag)
                exif_data[tag_name] = value
    return exif_data