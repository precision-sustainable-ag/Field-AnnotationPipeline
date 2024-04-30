# Field-AnnotationPipeline
This repo automates semantic labels and bounding boxes for Field imagery, within the broader Ag Image Repository, focusing on real-world agricultural conditions essential for training deep learning models.

This repository is dedicated to the development of a specialized image processing pipeline, designed exclusively for handling field imagery within the broader Ag Image Repository. It includes detailed metadata related to agricultural environments such as weeds, crops, and cover crops. Our pipeline focuses on automating the creation of semantic labels and bounding box annotations while leveraging extensive environmental metadata to enhance accuracy. Focused solely on images captured directly in agricultural fields, this project provides real-world conditions that are essential for effectively training deep learning models.

### Setting Up Your Environment Using an Environment File
After installing Conda, you can set up an environment for this project using an environment file, which specifies all necessary dependencies. Here's how:

1. Clone this repository to your local machine.
2. Navigate to the repository directory in your terminal.
3. Locate the `environment.yaml` file in the repository. This file contains the list of packages needed for the project.
4. Create a new Conda environment by running the following command:
   ```bash
   conda env create -f environment.yaml
   ```
   This command reads the `environment.yaml` file and creates an environment with the name and dependencies specified within it.

5. Once the environment is created, activate it with:
   ```bash
   conda activate <env_name>
   ```
   Replace `<env_name>` with the name of the environment specified in the `environment.yaml` file.