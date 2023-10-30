# DeNoise

[![Actions Status](https://github.com/ludgerradke/DeNoise/actions/workflows/BM3D.yml/badge.svg)](https://github.com/ludgerradke/DeNoise/actions/workflows/BM3D.yml/badge.svg)
[![Actions Status](https://github.com/ludgerradke/DeNoise/actions/workflows/CEST.yml/badge.svg)](https://github.com/ludgerradke/DeNoise/actions/workflows/CEST.yml/badge.svg)
[![Actions Status](https://github.com/ludgerradke/DeNoise/actions/workflows/DeepDenoise.yml/badge.svg)](https://github.com/ludgerradke/DeNoise/actions/workflows/DeepDenoise.yml/badge.svg)
[![Actions Status](https://github.com/ludgerradke/DeNoise/actions/workflows/NLM.yml/badge.svg)](https://github.com/ludgerradke/DeNoise/actions/workflows/NLM.yml/badge.svg)
[![Actions Status](https://github.com/ludgerradke/DeNoise/actions/workflows/PCA.yml/badge.svg)](https://github.com/ludgerradke/DeNoise/actions/workflows/PCA.yml/badge.svg)

This GitHub repository contains several Python packages that provide various functions in the field of image processing. The available packages are:

- **'BM3D'**: An implementation of the BM3D (Block Matching and 3D Filtering) noise reduction algorithm for 2D and 3D images.
- **'NLM'**: An implementation of the NLM (Non-Local Means) noise reduction algorithm for 2D images.
- **'PCA'**: Functions for applying PCA (Principal Component Analysis) to image data.
- **'DeepDenoise'**: Functions for applying deep neural networks to denoise images.
- **'Loader'**: Functions for loading image data.
- **'Metrics'**: Functions for calculating image metrics.
- **'CEST'**: Functions for processing CEST (Chemical Exchange Saturation Transfer) data.
- **'test_support_function'**: Support functions for testing the other packages.
- **'Transform'**: Functions for performing image transformations.

### **Related Publications:**
Our work on CEST Denosing is detailed in our recent publication. For an in-depth understanding and insights, please refer to:

- Karl Ludger Radke, Benedikt Kamp, Vibhu Adriaenssens, Julia Stabinska, Patrik Gallinnis, Hans-Jörg Wittsack, Gerald Antoch, and Anja Müller-Lutz. "Deep Learning-Based Denoising of CEST MR Data: A Feasibility Study on Applying Synthetic Phantoms in Medical Imaging." Diagnostics 2023, 13(21), 3326. DOI: [https://doi.org/10.3390/diagnostics13213326](https://doi.org/10.3390/diagnostics13213326)

***If you utilize the code or data provided in this repository, please cite our work in your publications. This will help in acknowledging our efforts and supporting the open science movement.***


## Installation

To install the repository, clone it from GitHub and install the required dependencies using pip:

````bash
git clone https://github.com/ludgerradke/DeNoise
pip install -r requirements.txt
````
To use one of the packages in the repository, import it into your Python code as follows:

````python
import BM3D
import CEST
import DeepDenoise
import Loader
import Metrics
import NLM
import PCA
import test_support_function
import Transform
````
For more information on how to use each package, see the corresponding ReadMe files.

## Denoising Command Line Interface (CLI)

This command-line interface (CLI) allows you to denoise NIFTI images using the BM3D, NLM, or PCA algorithms.

### Usage

To use the CLI, you will need to install the required dependencies and have Python 3.6 or later installed. You can then run the CLI by calling python denoise_cli.py from the command line, followed by the required arguments.

The required arguments are:

- **'input_path'**: the path to the input NIFTI image that you want to denoise
- **'output_path'**: the path where you want to save the denoised NIFTI image
- **'algorithm'**: the denoising algorithm to use. Must be one of **'BM3D'**, **'NLM'**, or **'PCA'**.

You can also specify the following optional arguments:

- **'--mask_path'**: the path to a mask NIFTI image, which can be used to denoise only certain regions of the input image
- **'--config'**: the path to a YAML file containing configuration parameters for the denoising algorithm. If no path is specified, the default configuration will be used.

### Configuration

The CLI allows you to specify configuration parameters for the denoising algorithms using a YAML file. The structure of this file will depend on the chosen algorithm and the parameters it supports.

By default, the CLI will use the following configuration files for each algorithm:

- **'default_yamls/bm3d.yaml'** for BM3D
- **'default_yamls/nlm.yaml'** for NLM
- **'default_yamls/pca.yaml'** for PCA

You can override these default configurations by specifying your own YAML file using the **'--config'** argument.

### Example

To denoise an image using the NLM algorithm and the default configuration, you can run the following command:

````bash
python denoise_cli.py input.nii.gz output.nii.gz NLM
````

To denoise an image using the BM3D algorithm and a custom configuration file, you can run the following command:

````bash
python denoise_cli.py input.nii.gz output.nii.gz BM3D --config custom_config.yaml
````


## Comparison of the denoising performance

Coming soon


## Support
If you really like this repository and find it useful, please consider (★) starring it, so that it can reach a broader audience of like-minded people.
