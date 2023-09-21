from typing import Optional, Tuple

import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn


class Noiser:
    def __init__(self, sigma: float = 0.01):
        """
        Initializes the Noiser object with the given sigma value.

        Args:
            sigma (float, optional): The sigma value for noise generation. Defaults to 0.01.
        """
        self.sigma = sigma

    def set_sigma(self, sigma: float) -> None:
        """
        Set the sigma value.

        Args:
            sigma (float): The sigma value to set.
        """
        self.sigma = sigma

    def add_noise(self, imgs: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """
        Adds white noise to the given images. If sigma is not provided, the sigma value
        of the Noiser object is used.

        Args:
            imgs (np.ndarray): The images to which noise will be added.
            sigma (float, optional): The sigma value for noise generation. Defaults to None.

        Returns:
            np.ndarray: The images with added noise.
        """
        if sigma is None:
            sigma = self.sigma
        if len(imgs.shape) == 3:
            for img_idx in range(imgs.shape[-1]):
                img = imgs[:, :, img_idx]
                noise_img = self.add_noise_in_k_space(img.shape, sigma)
                imgs[:, :, img_idx] = img + noise_img
        else:
            noise_img = self.add_noise_in_k_space(imgs.shape, sigma)
            imgs += noise_img
        return imgs

    def add_noise_in_k_space(self, shape: Tuple[int, ...], sigma: float) -> np.ndarray:
        """
        Generates rician noise with the given shape and sigma value in k-space.

        Args:
            shape (tuple): The shape of the white noise to be generated.
            sigma (float): The sigma value for noise generation.

        Returns:
            np.ndarray: The generated white noise.
        """
        dummy_img = np.zeros(shape)
        k_space_dummy = self.__transform_image_to_kspace(dummy_img)
        k_space_noise = self.__add_gaussian_noise(k_space_dummy, 1)
        noise_img = self.__transform_kspace_to_image(k_space_noise)
        return sigma * noise_img / noise_img.std()

    def __add_gaussian_noise(self, img: np.ndarray, sigma: float) -> np.ndarray:
        mean = 0
        shape = img.shape
        real_noise = np.random.normal(mean, sigma, shape)
        imag_noise = np.random.normal(mean, sigma, shape) * 1j
        return img + real_noise + imag_noise

    def __transform_kspace_to_image(
        self,
        k: np.ndarray,
        dim: Optional[np.ndarray] = None,
        img_shape: Optional[Tuple[int, ...]] = None,
    ) -> np.ndarray:
        if dim is None:
            dim = range(k.ndim)
        return np.abs(ifftn(ifftshift(k), s=img_shape, axes=dim))

    def __transform_image_to_kspace(
        self,
        img: np.ndarray,
        dim: Optional[np.ndarray] = None,
        k_shape: Optional[Tuple[int, ...]] = None,
    ) -> np.ndarray:
        if dim is None:
            dim = range(img.ndim)
        return fftshift(fftn(img, s=k_shape, axes=dim), axes=dim)
