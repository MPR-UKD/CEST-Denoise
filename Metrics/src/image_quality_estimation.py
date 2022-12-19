from abc import ABC

import numpy as np
import math


class IQS(ABC):
    # CEST Z-Spectra pixel_max = 1 / RGB Image 255
    def __init__(self, pixel_max: float | int, ref_image: np.ndarray | None = None):
        self.PIXEL_MAX = pixel_max
        self.ref_image = ref_image

    def mse(self,
            img1,
            img2: np.ndarray | None = None,
            mask: np.ndarray | None = None):
        if img2 is None:
            img2 = self.ref_image
        if mask is not None:
            img1 = img1[mask == 1]
            img2 = img2[mask == 1]
        mse = np.mean((img1 - img2) ** 2)
        return mse

    def psnr(self,
             img1: np.ndarray,
             img2: np.ndarray | None = None,
             mask: np.ndarray | None = None):
        if img2 is None:
            img2 = self.ref_image
        mse = self.mse(img1, img2, mask)
        if mse == 0:
            return 100
        return 20 * math.log10(self.PIXEL_MAX / math.sqrt(mse))

    def root_mean_square_error(self,
                               img1: np.ndarray,
                               img2: np.ndarray | None = None,
                               mask=None):
        if img2 is None:
            img2 = self.ref_image
        if mask is not None:
            img1 = img1[mask == 1]
            img2 = img2[mask == 1]
        return np.sqrt(((img1 - img2) ** 2).mean())