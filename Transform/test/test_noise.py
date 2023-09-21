import numpy as np
import pytest

from Transform.src.noise import Noiser


def test_noiser_init():
    # Test if the Noiser object is initialized with the correct sigma value
    noiser = Noiser(sigma=0.1)
    assert noiser.sigma == 0.1

    noiser = Noiser()
    assert noiser.sigma == 0.01


def test_noiser_set_sigma():
    # Test if the sigma value of the Noiser object is updated correctly
    noiser = Noiser()
    noiser.set_sigma(0.5)
    assert noiser.sigma == 0.5


def test_add_noise():
    # Test if the add_noise function correctly adds noise to the given images
    noiser = Noiser(sigma=0.1)

    # Test for a single image
    img = np.zeros((10, 10))
    noise_img = noiser.add_noise(img)
    assert noise_img.shape == img.shape
    assert np.abs(noise_img.std() - 0.1) < 1e-6

    # Test for multiple images
    imgs = np.zeros((10, 10, 3))
    noise_imgs = noiser.add_noise(imgs)
    assert noise_imgs.shape == imgs.shape
    assert np.abs(noise_imgs[:, :, 0].std() - 0.1) < 1e-6
    assert np.abs(noise_imgs[:, :, 1].std() - 0.1) < 1e-6
    assert np.abs(noise_imgs[:, :, 2].std() - 0.1) < 1e-6


def test_get_white_noise():
    # Test if the get_white_noise function correctly generates white noise with the given shape and sigma value
    noiser = Noiser(sigma=0.1)

    noise = noiser.add_noise_in_k_space((10, 10), 0.5)
    assert noise.shape == (10, 10)
    assert np.abs(noise.std() - 0.5) < 1e-6


if __name__ == "__main__":
    pytest.main()
