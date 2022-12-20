import numpy as np
from NLM.src.denoise import (
    nlm,
    nlm_CEST,
    pad_image,
    get_comparison_neighborhood,
    get_small_neighborhood,
)
from Metrics.src.image_quality_estimation import IQS
import pytest
from test_support_function.CEST import generate_Z_3D


def test_pad_image():
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    padded_image = pad_image(image, 2, 1)
    expected_padded_image = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 2, 3, 0],
            [0, 4, 5, 6, 0],
            [0, 7, 8, 9, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_equal(padded_image, expected_padded_image)


def test_get_comparison_neighborhood():
    padded_image = np.array(
        [[0, 1, 2, 3, 4],
         [5, 6, 7, 8, 9],
         [9, 10, 11, 12, 13],
         [14, 15, 16, 17, 18],
         [19, 20, 21, 22, 23]]
    )
    comparison_neighborhood = get_comparison_neighborhood(padded_image, 1, 1, 1)
    expected_comparison_neighborhood = np.array([[0, 1, 2], [5, 6, 7], [9, 10, 11]])
    np.testing.assert_equal(comparison_neighborhood, expected_comparison_neighborhood)


def test_get_small_neighborhood():
    padded_image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    small_neighborhood = get_small_neighborhood(padded_image, 0, 0, 1)
    expected_small_neighborhood = np.array([[1, 2], [4, 5]])
    np.testing.assert_equal(small_neighborhood, expected_small_neighborhood)


def test_nlm():
    image = np.ones((42, 42)) * 125
    noise = image + np.random.randint(0, 5, (42, 42))
    denoise = nlm(noise.copy(), 22, 6)
    iqs = IQS(pixel_max=255, ref_image=image)
    assert iqs.psnr(denoise) > iqs.psnr(noise)


def test_nlm_CEST():
    iqs = IQS(pixel_max=1)
    Z = generate_Z_3D(img_size=(42, 42), dyn=5, ppm=3, a=1, b=-0.5, c=3)
    Z_noisy = Z + 0.2 * np.random.random(Z.shape)

    Z_denoise = nlm_CEST(Z, 22, 6)

    assert Z.shape == Z_denoise.shape
    assert iqs.psnr(Z_denoise[:, :, 0], Z[:, :, 0]) > iqs.psnr(
        Z_denoise[:, :, 0], Z_noisy[:, :, 0]
    )

def test_nlm_CEST_ml():
    iqs = IQS(pixel_max=1)
    Z = generate_Z_3D(img_size=(42, 42), dyn=5, ppm=3, a=1, b=-0.5, c=3)
    Z_noisy = Z + 0.2 * np.random.random(Z.shape)

    Z_denoise = nlm_CEST(Z, 22, 6, multi_processing=True)

    assert Z.shape == Z_denoise.shape
    assert iqs.psnr(Z_denoise[:, :, 0], Z[:, :, 0]) > iqs.psnr(
        Z_denoise[:, :, 0], Z_noisy[:, :, 0]
    )


if __name__ == "__main__":
    pytest.main()
