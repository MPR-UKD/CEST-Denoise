# BM3D Denoising
This package provides an implementation of the BM3D (Block-Matching and 3D Filtering) denoising algorithm for 2D and 3D images. The main function for denoising is bm3d, which takes an input image and a set of configuration parameters and returns the denoised image.

## Functionality
The main function for denoising is **'bm3d(img, config=None, mask=None)'**, which takes the following inputs:

- **img**: The input image to be denoised, which should be a 2D or 3D numpy array.
- **config**: An optional dictionary of configuration parameters for the BM3D denoising process. If not provided, default values will be used.
- **mask**: An optional mask that specifies which pixels in the image should be denoised. This should be a 2D or 3D numpy array of the same size as **'img'**.

The function returns the denoised image, which is a numpy array of the same size as the input image.

In addition to the **bm3d** function, there is also a function **'bm3d_CEST(img, mask=None, config=None, multi_processing=False)'** that can be used to denoise CEST (Chemical Exchange Saturation Transfer) data. This function works in the same way as the **'bm3d'** function, but it takes an additional input multi_processing, which specifies whether to use multi-processing to denoise the image. Furthermore the function converts the input image to an integer array, with values in the range [0, 255]. It then performs the BM3D denoising process either on all slices of the image (if multi_processing is False) or on each slice in parallel (if multi_processing is True). The function returns the denoised image, which is scaled back to the range [0, 1].

## Configuration Parameters

The following configuration parameters can be specified in the **'config'** dictionary:

- **'sigma'**: The standard deviation of the Gaussian noise present in the image.
- **'lamb2d'**: A parameter that controls the strength of the 2D transformation applied to the blocks in the first step of the denoising process.
- **'lamb3d'**: A parameter that controls the strength of the 3D transformation applied to the blocks in the second step of the denoising process.
- **'KaiserWindowBeta'**: A parameter that controls the shape of the Kaiser window used in the block matching process.
- **'TransformationFunction'**: The discrete transformation function to be used in the block matching process. This can be either "cos" or "sin".
- **'step1_threshold_distance'**: The threshold distance used in the first step of the denoising process.
- **'step1_max_match'**: The maximum number of matched blocks used in the first step of the denoising process.
- **'step1_BlockSize'**: The block size used in the first step of the denoising process.
- **'step1_WindowSize'**: The search window size used in the first step of the denoising process.
- **'step2_threshold_distance'**: The threshold distance used in the second step of the denoising process.
- **'step2_max_match'**: The maximum number of matched blocks used in the second step of the denoising process.
- **'step2_BlockSize'**: The block size used in the second step of the denoising process.
- **'step2_WindowSize'**: The search window size used in the second step of the denoising process.

## Example

Here is an example of how to use the bm3d function to denoise a 2D image:

````python
import numpy as np
from BM3D.src.denoising import bm3d

# Load the noisy image as a numpy array
noisy_image = np.load('noisy_image.npy')

# Set the configuration parameters
config = {'sigma': 25, 'lamb2d': 2, 'lamb3d': 2.7, 'KaiserWindowBeta': 2, 'TransformationFunction': 'cos'}

# Denoise the image
denoised_image = bm3d(noisy_image, config)

# Save the denoised image
np.save('denoised_image.npy', denoised_image)
````

## Contribution

This **BM3D** denoising project is open source, and anyone is welcome to contribute to it. If you have an idea for a new feature or have found a bug, you can create a pull request or open an issue on the project's GitHub page.

Before making any changes to the code, it is recommended to ensure that the pytests in the **'BM3D/test'** directory are still working as expected. This will help to ensure that the changes you are making do not break any existing functionality.

We encourage any and all contributions to this project, and we appreciate your help in making it better for everyone.