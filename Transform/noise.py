import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn


class Noiser:
    def __init__(self, sigma: float = 0.01):
        self.sigma = sigma

    def set_sigma(self, sigma: float):
        self.sigma = sigma

    def add_noise(self, imgs: np.ndarray, sigma: float | None = None) -> np.ndarray:
        if sigma is None:
            sigma = self.sigma
        if len(imgs.shape) == 3:
            for img_idx in range(imgs.shape[-1]):
                img = imgs[:, :, img_idx]
                noise_img = self.get_white_noise(img.shape, sigma)
                imgs[:, :, img_idx] = img + noise_img
        else:
            noise_img = self.get_white_noise(imgs.shape, sigma)
            imgs += noise_img
        return imgs

    def get_white_noise(self, shape: tuple, sigma: float) -> np.ndarray:
        dummy_img = np.zeros(shape)
        k_space_dummy = self.__transform_image_to_kspace(dummy_img)
        k_space_noise = self.__add_gaussian_noise(k_space_dummy, 1)
        noise_img = self.__transform_kspace_to_image(k_space_noise).real
        return sigma * noise_img / noise_img.std()

    def __add_gaussian_noise(self, img: np.ndarray, sigma: float) -> np.ndarray:
        mean = 0
        shape = img.shape
        return img + np.random.normal(mean, sigma, shape)

    def __transform_kspace_to_image(
        self,
        k: np.ndarray,
        dim: np.ndarray | None = None,
        img_shape: tuple | None = None,
    ) -> np.ndarray:
        """Computes the Fourier transform from k-space to image space
        along a given or all dimensions
        :param k: k-space data
        :param dim: vector of dimensions to transform
        :param img_shape: desired shape of output image
        :returns: data in image space (along transformed dimensions)
        """
        if not dim:
            dim = range(k.ndim)
        return ifftn(ifftshift(k), s=img_shape, axes=dim)

    def __transform_image_to_kspace(
        self,
        img: np.ndarray,
        dim: np.ndarray | None = None,
        k_shape: tuple | None = None,
    ) -> np.ndarray:
        """Computes the Fourier transform from image space to k-space space
        along a given or all dimensions
        :param img: image space data
        :param dim: vector of dimensions to transform
        :param k_shape: desired shape of output k-space data
        :returns: data in k-space (along transformed dimensions)
        """
        if not dim:
            dim = range(img.ndim)
        return fftshift(fftn(img, s=k_shape, axes=dim), axes=dim)
