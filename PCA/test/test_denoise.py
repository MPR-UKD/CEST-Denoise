from Metrics.src.image_quality_estimation import IQS
import pytest
from test_support_function.src.CEST import generate_Z_3D
from PCA.src.denoise import pca
import numpy as np


def test_median():
    iqs = IQS(pixel_max=1)
    Z = generate_Z_3D(img_size=(42, 42), dyn=5, ppm=3, a=1, b=-0.5, c=3)
    Z_noisy = Z + 0.2 * np.random.random(Z.shape)

    Z_denoise = pca(Z, "median")

    assert Z.shape == Z_denoise.shape
    assert iqs.psnr(Z_denoise[:, :, 0], Z[:, :, 0]) > iqs.psnr(
        Z_noisy[:, :, 0], Z[:, :, 0]
    )


def test_nelson():
    iqs = IQS(pixel_max=1)
    Z = generate_Z_3D(img_size=(42, 42), dyn=5, ppm=3, a=1, b=-0.5, c=3)
    Z_noisy = Z + 0.2 * np.random.random(Z.shape)

    Z_denoise = pca(Z, "nelson")

    assert Z.shape == Z_denoise.shape
    assert iqs.psnr(Z_denoise[:, :, 0], Z[:, :, 0]) > iqs.psnr(
        Z_noisy[:, :, 0], Z[:, :, 0]
    )


def test_malinowski():
    iqs = IQS(pixel_max=1)
    Z = generate_Z_3D(img_size=(42, 42), dyn=5, ppm=3, a=1, b=-0.5, c=3)
    Z_noisy = Z + 0.2 * np.random.random(Z.shape)

    Z_denoise = pca(Z, "malinowski")

    assert Z.shape == Z_denoise.shape
    assert iqs.psnr(Z_denoise[:, :, 0], Z[:, :, 0]) > iqs.psnr(
        Z_noisy[:, :, 0], Z[:, :, 0]
    )


if __name__ == "__main__":
    pytest.main()
