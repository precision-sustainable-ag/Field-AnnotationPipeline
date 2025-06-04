
# Image Acquisition and Packaging for CVAT

This repository contains two scripts for preparing species-specific image datasets for CVAT annotation.

---

## Overview

1. **`acquire_image_from_lts.py`**  
   Randomly selects images for each species from long-term storage and copies them to a structured directory.

2. **Manual Step**  
   After running the acquire script:
   - If enough cutouts exists: **manually select 5 good cutout images and masks** for each species and then proceed to next step.
   - If cutouts do not exists: run the segmentation pipeline to get the cutouts and then proceed to next step. 

3. **`package_cvat.py`**  
   Packages the selected images and their masks into a CVAT-compatible zipped dataset.

---

## Pipeline Steps

### 1. Acquire random images from LTS

**Script:** `acquire_image_from_lts.py`

- **Purpose:**  
  Selects 20 random images per species from available cutouts or developed images.
  
- **Usage:**  
  ```bash
  python acquire_image_from_lts.py
  ```

- **Output:**  
  Images organized by species inside:
  ```
  /field-tools/field_test_dataset/species_dataset/{species_name}/cutouts/
  or
  /field-tools/field_test_dataset/species_dataset/{species_name}/developed-images/
  ```

---

### 2. Manually choose best 5 cutouts

- Navigate to the `cutouts` folder for each species.
- **Manually select** the **5 best quality cutout images**.
- Remove any extra cutouts you don't want to include.

> **Note:**  
> Only the selected 5 images and their corresponding masks will be packaged later.

---

### 3. Package images and masks for CVAT

**Script:** `package_cvat.py`

- **Purpose:**  
  Converts selected cutout images and corresponding masks into a CVAT-compatible zipped package (image/mask mapping + color labels).
  
- **Usage:**  
  ```bash
  python package_cvat.py
  ```

- **Output:**  
  Zipped package at:
  ```
  /field-tools/field_test_dataset/species_dataset/{species_name}_cvat_package.zip
  ```

Each ZIP file contains:
- Images
- 3-channel masks
- Image-to-mask mapping `.txt` file
- Color label file (`label_colors.txt`) with:
  ```
  0 0 0 background
  255 255 255 weed
  ```
