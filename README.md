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

1. **Detect Weeds**:
    ```bash
    python FIELD_PIPELINE.py
    ```

For more detailed instructions, refer to the inline comments within the scripts.

## Scripts

- `FIELD_PIPELINE.py`: This script executes the pipeline tasks defined in the configuration file.
- `detect_weeds.py`: This script detects weeds in field images and creates bounding box annotations.
- `segment_weeds.py`: This script segments weeds and generates semantic labels.

## Configuration

The pipeline is configured using [HYDRA](https://github.com/facebookresearch/hydra.git). This file contains parameters that control various aspects of the pipeline, such as input directories, output formats, and processing parameters.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


