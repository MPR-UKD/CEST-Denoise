# NLM Denoising

This package provides an implementation of the Non-Local Means (NLM) denoising algorithm for 2D and 3D images. The main function for denoising is **'nlm'**, which takes an input image, a large window size, and a small window size and returns the denoised image.

## Functionality
The main function for denoising is **'nlm(image, big_window_size, small_window_size)'**, which takes the following inputs:

- **'image'**: The input image to be denoised, which should be a 2D or 3D numpy array.
- **'big_window_size'**: The size of the large window that will be used to compare neighborhoods in the denoising process.
- **'small_window_size'**: The size of the small window that will be used to extract neighborhoods from the image.

The function returns the denoised image, which is a 2D numpy array of the same size as the input image.

There is also a function **'nlm_CEST'** that can be used to denoise that can be used to denoise CEST (Chemical Exchange Saturation Transfer) data. This function works in the same way as the **'nlm'** function, but it takes an additional input multi_processing, which specifies whether to use multi-processing to denoise the image as well as an additional input pools that specifies the number of pools to use for the multi-processing. Furthermore the function converts the input image to an integer array, with values in the range [0, 255]. It then performs the NLM denoising process either on all slices of the image (if multi_processing is False) or on each slice in parallel (if multi_processing is True). The function returns the denoised image, which is scaled back to the range [0, 1].

The NLM denoising algorithm works by comparing each pixel in the image to its neighbors within a specified search window, and weighting the contribution of each neighbor based on the distance between them. The final denoised pixel value is the weighted average of all the neighbor values.

## Example

````python
import numpy as np
from NLM.src.denoising import nlm

# Load the noisy image as a numpy array
noisy_image = np.load('noisy_image.npy')

# Set the configuration parameters
big_window_size = 7
small_window_size = 3

# Denoise the image
denoised_image = nlm(noisy_image, big_window_size, small_window_size)
````

## Performance

The NLM denoising algorithm can be computationally expensive, especially when applied to large images. It is recommended to experiment with different window sizes to find the best trade-off between denoising performance and computational time.

## Contribution

This **NLM** denoising project is open source, and anyone is welcome to contribute to it. If you have an idea for a new feature or have found a bug, you can create a pull request or open an issue on the project's GitHub page.

Before making any changes to the code, it is recommended to ensure that the pytests in the **'NLM/test'** directory are still working as expected. This will help to ensure that the changes you are making do not break any existing functionality.

We encourage any and all contributions to this project, and we appreciate your help in making it better for everyone.