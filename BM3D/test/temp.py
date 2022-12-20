import numpy as np
#import pytest

from BM3D.src.denoise import bm3d

"""
def test_bm3d():
    # Create mock input data
    img = (np.zeros((42, 42)) * 255).astype("int16")
    config = {
        "sigma": 25.0,
        "lamb2d": 2.0,
        "lamb3d": 2.7,
        "KaiserWindowBeta": 2.0,
        "TransformationFunction": "cos",
        "step1_threshold_distance": 2500,
        "step1_max_match": 16,
        "step1_BlockSize": 8,
        "step1_WindowSize": 39,
        "step2_threshold_distance": 400,
        "step2_max_match": 32,
        "step2_BlockSize": 8,
        "step2_WindowSize": 39,
    }
    mask = np.ones((42, 42))

    # Test 1 - check that the function returns an image of the correct shape
    final_img = bm3d(img, config, mask)
    assert final_img.shape == (42, 42)

    # Test 2 - Check if the denoised image has less noise than the original noisy image
    noise_level = 25
    img_shape = (42, 42)
    np.random.seed(0)
    noisy_image = (np.random.randint(0, 255, (42,42))).astype('int16')

    # Denoise the image
    denoised_image = bm3d(noisy_image, config={"sigma": noise_level})

    assert np.std(denoised_image) < np.std(noisy_image)

    # Test 3 - check that the function handles different block sizes correctly
    config = {"sigma": 25.0, "step1_BlockSize": 4, "step2_BlockSize": 4}
    result = bm3d(img, config, mask)
    assert result.shape == (42, 42)

    config = {"sigma": 25.0, "step1_BlockSize": 16, "step2_BlockSize": 16}
    result = bm3d(img, config, mask)
    assert result.shape == (42, 42)

    # Test 4 - check that the function handles different window sizes correctly
    config = {"sigma": 25.0, "step1_WindowSize": 21, "step2_WindowSize": 21}
    result = bm3d(img, config, mask)
    assert result.shape == (42, 42)

    config = {"sigma": 25.0, "step1_WindowSize": 41, "step2_WindowSize": 41}
    result = bm3d(img, config, mask)
    assert result.shape == (42, 42)

    # Test 5 - check that the function handles different sigma values correctly
    config = {"sigma": 10.0}
    result = bm3d(img, config, mask)
    assert result.shape == (42, 42)

    config = {"sigma": 50.0}
    result = bm3d(img, config, mask)
    assert result.shape == (42, 42)

    # Test 6 - check that the function handles different Kaiser window beta values correctly
    config = {"sigma": 25.0, "KaiserWindowBeta": 1.0}
    result = bm3d(img, config, mask)
    assert result.shape == (42, 42)

    config = {"sigma": 25.0, "KaiserWindowBeta": 4.0}
    result = bm3d(img, config, mask)
    assert result.shape == (42, 42)

    # Test 7 - check that the function handles different transformation functions correctly
    config = {"sigma": 25.0, "TransformationFunction": "cos"}
    result = bm3d(img, config, mask)
    assert result.shape == (42, 42)

    config = {"sigma": 25.0, "TransformationFunction": "sin"}
    result = bm3d(img, config, mask)
    assert result.shape == (42, 42)
"""


if __name__ == '__main__':
    img = (np.zeros((42, 42)) * 255).astype("int16")
    mask = np.ones((42, 42))
    config = {"sigma": 25.0, "step1_WindowSize": 41, "step2_WindowSize": 41}
    result = bm3d(img, config, mask)
    b = 2
    #pytest.main()
