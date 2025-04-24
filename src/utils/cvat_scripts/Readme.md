# CVAT Image Packaging Pipeline

These two scripts select and prepare images for annotation in CVAT.

## 1. Acquire Images

**Script:** `acquire_images_from_lts.py`  
Selects random `.jpg` images by species from LTS for annotation.

```bash
python acquire_images_from_lts.py
```

## 2. Run Annotation Pipeline

Use the `field_annotation_pipeline` to generate cutouts and predicted masks from the selected images.

## 3. Copy Processed Files

Move the created species-wise directories into the appropriate folders in the LTS directory located at: /mnt/research-projects/r/raatwell/longterm_images3/field-tools/field_test_dataset

## 4. Package for CVAT

**Script:** `package_cvat.py`  
Packages images and masks in a CVAT-compatible format.

```bash
python package_cvat.py
```

**Output:** A ready-to-upload CVAT zip with images, 3-channel masks, and metadata.