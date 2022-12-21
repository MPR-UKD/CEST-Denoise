# Metrics

This repository contains a package for calculating various metrics. The package includes functions for calculating Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), Root Mean Squared Error (RMSE), and Magnetization Transfer Ratio (MTR) asymmetry.

## Usage

The package includes the following functions:

**'mse'**

Calculates the MSE between two images.

````python
from Metrics.src.image_quality_estimation import IQS

# Create an IQS object
iqs = IQS(pixel_max=255)

# Calculate the MSE between two images
mse = iqs.mse(img1, img2)
````

**'psnr'**

Calculates the PSNR between two images.

````python
from Metrics.src.image_quality_estimation import IQS

# Create an IQS object
iqs = IQS(pixel_max=255)

# Calculate the PSNR between two images
mse = iqs.psnr(img1, img2)
````

**'root_mean_square_error'**

Calculates the RMSE between two images.

````python
from Metrics.src.image_quality_estimation import IQS

# Create an IQS object
iqs = IQS(pixel_max=255)

# Calculate the RMSE between two images
mse = iqs.root_mean_square_error(img1, img2)
````

**'mtr_asym'**

Calculates MTR asymmetry and MTR asymmetry image of CEST data.

````python
from Metrics.src.CEST import mtr_asym

# Calculate MTR asymmetry and MTR asymmetry image
# - cest_data = np.array with shape (x, y, dyn)
# - mask = binary np.array with shape (x, y)
# - mtr_asym_ppm = tuple with mtr_asym calculation range (lower, upper)
# - ppm = float with ppm range
mtr_asym, mtr_asym_img = mtr_asym(cest_data, mask, mtr_asym_ppm, ppm)
````