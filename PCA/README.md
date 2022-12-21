<span style="color:red; font-size:24px">Please note that this method works only for CEST data</span>

# PCA Denoising for CEST MRI

This is an implementation of a method for denoising chemical exchange saturation transfer (CEST) MRI images using principal component analysis (PCA). The method involves fitting a model to the CEST MRI data and using it to estimate the noise in the data, which is then used to improve the signal-to-noise ratio (SNR) of the CEST MRI images.

## Functionality

The main function for denoising is **'pca(img, criteria, mask=None)'**, which takes the following inputs:

- **img**: The input image to be denoised, which should be a 2D numpy array with dimensions (x, y, ndyn).
- **criteria**: The criteria to use for determining the optimal number of components. Can be one of 'malinowski', 'nelson', or 'median'. In addition, an integer can also be passed directly, in which case no criterion is used but the number of components passed.
- **mask**: An optional binary mask that specifies which pixels in the image should be denoised. This should be a 2D numpy array of the same size as 'img'.

The function returns the denoised image as a 2D numpy array.

## Example

This will denoise the example CEST MRI image using the Malinowski criteria and plot the original and denoised images.
````python
import numpy as np
from pca_denoising import pca

# Load CEST MRI image and mask
img = np.load('cest_image.npy')
mask = np.load('mask.npy')

# Denoise the image using PCA and the Malinowski criteria
denoised_img = pca(img, criteria='malinowski', mask=mask)
````

## Contribution

This **PCA** denoising project is open source, and anyone is welcome to contribute to it. If you have an idea for a new feature or have found a bug, you can create a pull request or open an issue on the project's GitHub page.

Before making any changes to the code, it is recommended to ensure that the pytests in the **'PCA/test'** directory are still working as expected. This will help to ensure that the changes you are making do not break any existing functionality.

We encourage any and all contributions to this project, and we appreciate your help in making it better for everyone.


## References

Breitling, J., Deshmane, A., Goerke, S., Korzowski, A., Herz, K., Ladd, M. E., Scheffler, K., Bachert, P., & Zaiss, M. (n.d.). Adaptive denoising for chemical exchange saturation transfer MR imaging.