from abc import ABC

import numpy as np
import math


class IQS(ABC):
    # CEST Z-Spectra pixel_max = 1 / RGB Image 255
    def __init__(self, pixel_max: float, ref_image: np.ndarray = None):
        # The maximum value that a pixel can take
        self.PIXEL_MAX = pixel_max
        # The reference image to compare against
        self.ref_image = ref_image

    def mse(self,
            img1,
            img2: np.ndarray = None,
            mask: np.ndarray = None):
        # If img2 is not provided, use the reference image
        if img2 is None:
            img2 = self.ref_image
        # If a mask is provided, use only the masked pixels for comparison
        if mask is not None:
            img1 = img1[mask == 1]
            img2 = img2[mask == 1]
        # Calculate the mean squared error between img1 and img2
        mse = np.mean((img1 - img2) ** 2)
        return mse

    def psnr(self,
             img1: np.ndarray,
             img2: np.ndarray = None,
             mask: np.ndarray = None):
        # If img2 is not provided, use the reference image
        if img2 is None:
            img2 = self.ref_image
        # Calculate the mean squared error between img1 and img2
        mse = self.mse(img1, img2, mask)
        # If the MSE is zero, the PSNR is infinite
        if mse == 0:
            return 100
        # Calculate the PSNR using the MSE and PIXEL_MAX
        return 20 * math.log10(self.PIXEL_MAX / math.sqrt(mse))

    def root_mean_square_error(self,
                               img1: np.ndarray,
                               img2: np.ndarray = None,
                               mask=None):
        # If img2 is not provided, use the reference image
        if img2 is None:
            img2 = self.ref_image
        # If a mask is provided, use only the masked pixels for comparison
        if mask is not None:
            img1 = img1[mask == 1]
            img2 = img2[mask == 1]
        # Calculate the root mean squared error between img1 and img2
        return np.sqrt(((img1 - img2) ** 2).mean())
