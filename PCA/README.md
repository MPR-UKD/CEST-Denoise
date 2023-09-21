# PCA Denoising for CEST MRI

## ⚠️ **Attention**
This method is specifically designed and validated **only for CEST data**. Application to other forms of data is not recommended without proper validation.

## Overview

This project is an implementation of a method for denoising Chemical Exchange Saturation Transfer (CEST) MRI images using Principal Component Analysis (PCA). It aims to enhance the signal-to-noise ratio (SNR) by estimating and mitigating the impact of noise in CEST MRI images.

## Features

### Main Function: `pca(img, criteria, mask=None)`
- **img**: 2D numpy array (x, y, ndyn). Represents the input image to be denoised.
- **criteria**: String or Integer. Defines the criteria to use for determining the optimal number of components. Acceptable string values are 'malinowski', 'nelson', or 'median'. If an integer is provided, it is used directly as the number of components.
- **mask**: Optional; 2D numpy array (x, y). A binary mask specifying which pixels in the image should be denoised.

**Returns**: The denoised image as a 2D numpy array.

## Quick Start

Here's a simple example demonstrating the usage of the `pca` function with the Malinowski criteria:

```python
import numpy as np
from pca_denoising import pca

# Load CEST MRI image and mask
img = np.load('cest_image.npy')
mask = np.load('mask.npy')

# Perform PCA denoising using the Malinowski criteria
denoised_img = pca(img, criteria='malinowski', mask=mask)
```

## Contribute

We warmly welcome and appreciate contributions from the community. If you have ideas for improvement, bug fixes, or new features, feel free to create a pull request or open an issue.

### Testing
Before contributing, please ensure that all pytests in the **'PCA/test'** directory pass to maintain the integrity and reliability of the code.

## References

- Breitling, J., Deshmane, A., Goerke, S., Korzowski, A., Herz, K., Ladd, M. E., Scheffler, K., Bachert, P., & Zaiss, M. (n.d.). Adaptive denoising for chemical exchange saturation transfer MR imaging.

