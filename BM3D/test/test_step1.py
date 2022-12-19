import pytest
import numpy as np

from BM3D.src.step1 import step1_basic_estimation


def test_step1_basic_estimation():
    # Create a test image
    noisy_img = np.random.rand(10, 10)

    # Set the parameters for the function
    param = (2, 1, 1, 3, 1, 1, 1, 1, "cos")
    mask = np.ones((10, 10))  # Mask with all pixels included

    # Run the function
    basic_estimate_img = step1_basic_estimation(noisy_img, param, mask)

    # Check that the returned image has the correct shape
    assert basic_estimate_img.shape == noisy_img.shape

    # Check that the sum of the pixels in the basic estimate image is not equal to the sum of the pixels in the noisy image
    # (assuming that the function is actually removing noise from the image)
    assert np.sum(basic_estimate_img) != np.sum(noisy_img)

    # Check that the sum of the pixels in the basic estimate image is not equal to 0 (assuming that the function is
    # returning a valid image)
    assert np.sum(basic_estimate_img) != 0


if __name__ == '__main__':
    pytest.main()
