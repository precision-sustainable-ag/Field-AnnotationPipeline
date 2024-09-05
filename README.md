# Field-AnnotationPipeline

This repository automates the creation of semantic labels and bounding boxes for field imagery within the broader Ag Image Repository, focusing on real-world agricultural conditions essential for training deep learning models.

## Introduction

The `Field-AnnotationPipeline` is a specialized image processing pipeline designed exclusively for handling field imagery within the Ag Image Repository. This pipeline automates the generation of semantic labels and bounding box annotations for various agricultural environments, including weeds, crops, and cover crops. By leveraging extensive environmental metadata, the pipeline enhances the accuracy of annotations, providing real-world conditions crucial for effectively training deep learning models.

## Installation

To set up the environment, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/precision-sustainable-ag/Field-AnnotationPipeline.git
    ```

2. Install dependencies using the provided `environment.yaml`:
    ```bash
    conda env create -f environment.yaml
    conda activate field-annotation-pipeline
    ```

## Usage

Once the environment is set up, you can start using the pipeline to annotate your field imagery.

5. Once the environment is created, activate it with:
   ```bash
   conda activate <env_name>
   ```
   Replace `<env_name>` with the name of the environment specified in the `environment.yaml` file.


## Find Unprocessed Batches

`find_unprocessed_batches` script assess the status of batches stored in long-term storage. It checks each batch for the presence of raw images, developed images, metadata, and cutouts, generating a table (known as "report") that provides insight into the completeness of each batch. The script saves this report as a CSV file, making it easy to review and analyze the data.

- **Content Verification**: Checks each batch for specific subdirectories (e.g., `raws`, `developed-images`, `metadata`, `cutouts`) and counts the files in each, ensuring that all necessary data is present.
- **Automated Reporting**: Generates a timestamped report summarizing the contents of each batch, which is saved as a CSV file.

### Configuration

The script requires a configuration file (`conf/config.yaml`) that specifies the paths for long-term storage and the report directory. Key configuration parameters include:

- `cfg.data.longterm_storage`: Path to the directory where the image batches are stored long-term.
- `cfg.reports`: Path to the directory where the generated CSV reports will be saved.


### Output

- **CSV Report**: The script generates a timestamped CSV report in the specified `reports` directory. The report includes the following columns for each batch:
  - `Batch`: The name of the batch folder.
  - `Raw Images`: The count of raw image files (e.g., `.ARW` files).
  - `Developed Images`: The count of developed image files (e.g., `.jpg` files).
  - `Metadata`: The count of metadata files (e.g., `.json` files).
  - `Cutouts`: The count of cutout image files (e.g., `.png` files).


## Download Batches

The `BatchDownloader` script is a component of the larger Field image annotation processing pipeline. This script automates the process of identifying and downloading batches of images that meet specific criteria from a long-term storage location to a temporary directory for further processing. The script ensures that only complete and necessary batches are downloaded, and it includes configurable options to manage the download process.

- **Batch Filtering:** Filters image batches based on specified criteria, such as having zero cutouts, zero metadata, and an equal number of raw and developed images.
- **Integrity Check:** Before downloading, checks whether the batch already exists in the temporary storage. If it does, the script verifies that all expected images are present. If any images are missing, the script removes the batch folder and re-downloads the entire batch.
- **Download Limiting:** Limits the number of batches downloaded in a single run, which is configurable through the pipeline's [config file](./conf/config.yaml). Takes into account batches that are already present in the temporary directory.
- **Logging:** The script logs all major actions and checks, including the number of images present in each batch versus the expected count.
- **Multithreading:** Multithreaded downloading of images within each batch. The number of threads is adjusted based on the system's available CPU cores.

### Configuration

The script relies on a configuration file (`conf/config.yaml`) for its operation. Key configuration parameters include:

- `data.longterm_storage`: Path to the directory where the image batches are stored long-term.
- `data.temp_dir`: Path to the temporary directory where batches are downloaded for processing.
- `download_batch.download_limit`: The maximum number of batches to download in a single run, includes batches that have already been downloaded.
- `download_batch.use_multithreading`: Enables or disables multithreading for image downloading within a batch.

TODO: add outputs for this script