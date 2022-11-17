import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn


class Noiser:

    def __init__(self):
        self.sigma = 0.01

    def set_sigma(self, sigma):
        self.sigma = sigma

    def add_noise(self, imgs):
        for img_idx in range(imgs.shape[-1]):
            img = imgs[:, :, img_idx]
            noise_img = self.get_white_noise(img.shape, self.sigma)
            imgs[:, :, img_idx] = img + noise_img
        return imgs

    def get_white_noise(self, shape, sigma):
        dummy_img = np.zeros(shape)
        k_space_dummy = self.__transform_image_to_kspace(dummy_img)
        k_space_noise = self.__add_gaussian_noise(k_space_dummy, 1)
        noise_img = self.__transform_kspace_to_image(k_space_noise).real
        return sigma * noise_img / noise_img.std()

    def __add_gaussian_noise(self, img, sigma):
        mean = 0
        shape = img.shape
        return img + np.random.normal(mean, sigma, shape)

    def __transform_kspace_to_image(self, k, dim=None, img_shape=None):
        """ Computes the Fourier transform from k-space to image space
            along a given or all dimensions
            :param k: k-space data
            :param dim: vector of dimensions to transform
            :param img_shape: desired shape of output image
            :returns: data in image space (along transformed dimensions)
            """
        if not dim:
            dim = range(k.ndim)

        # img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
        img = ifftn(ifftshift(k), s=img_shape, axes=dim)
        # img = ifftn(k, s=img_shape, axes=dim)
        # img *= np.sqrt(np.prod(np.take(img.shape, dim)))
        return img

    def __transform_image_to_kspace(self, img, dim=None, k_shape=None):
        """ Computes the Fourier transform from image space to k-space space
            along a given or all dimensions
            :param img: image space data
            :param dim: vector of dimensions to transform
            :param k_shape: desired shape of output k-space data
            :returns: data in k-space (along transformed dimensions)
            """
        if not dim:
            dim = range(img.ndim)

        # k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
        k = fftshift(fftn(img, s=k_shape, axes=dim), axes=dim)
        # k = fftn(img, s=k_shape, axes=dim)
        # k /= np.sqrt(np.prod(np.take(img.shape, dim)))
        return k
