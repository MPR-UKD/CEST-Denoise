import pytest
from BM3D.src.step2 import *


def test_step2_final_estimation():

    basic_estimate_img = np.zeros((42, 42))
    noisy_image = np.zeros((42, 42))
    param = (8, 16, 32, 32, 10, 1.5, "cos")
    mask = np.ones((42, 42))
    verbose = False

    # Test 1: Ensure the function returns an array of the correct shape
    final_img = step2_final_estimation(
        basic_estimate_img, noisy_image, param, mask, verbose
    )
    assert final_img.shape == (42, 42)

    # Test 2: Ensure the function returns the correct value when the mask is all zeros
    mask = np.zeros((42, 42))
    final_img = step2_final_estimation(
        basic_estimate_img, noisy_image, param, mask, verbose
    )
    assert np.all(final_img == 0)

    # Test 3: Ensure the function returns the correct value when the mask has only a single non-zero element
    basic_estimate_img[21, 21] = 1
    noisy_image[21, 21] = 1
    mask[21, 21] = 1
    final_img = step2_final_estimation(
        basic_estimate_img, noisy_image, param, mask, verbose
    )
    assert final_img[21, 21] > 0
    assert final_img[22, 22] < final_img[21, 21]

    # Test 4: Ensure the function returns the correct value when the noisy image is all ones
    basic_estimate_img = np.zeros((42, 42))
    noisy_image = np.ones((42, 42))
    mask = np.ones((42, 42))
    final_img = step2_final_estimation(
        basic_estimate_img, noisy_image, param, mask, verbose
    )
    assert np.all(final_img <= 1)


def test_grouping():
    # Create mock input data
    basic_estimate_img = np.zeros((42, 42))
    ref_block_position = (0, 0)
    block_size = 8
    threshold_distance = 16
    max_number_of_similar_blocks = 32
    window_size = 32
    discrete_transform_blocks_basic = np.zeros((42, 42, block_size, block_size))
    discrete_transform_blocks_noisy = np.zeros((42, 42, block_size, block_size))

    # Ensure the function returns the expected outputs
    block_positions, block_groups_basic, block_groups_noisy = grouping(
        basic_estimate_img,
        ref_block_position,
        block_size,
        threshold_distance,
        max_number_of_similar_blocks,
        window_size,
        discrete_transform_blocks_basic,
        discrete_transform_blocks_noisy,
    )
    assert block_positions.shape == (block_groups_noisy.shape[0], 2)
    assert block_groups_basic.shape == (
        max_number_of_similar_blocks,
        block_size,
        block_size,
    )
    assert block_groups_noisy.shape == (
        max_number_of_similar_blocks,
        block_size,
        block_size,
    )


def test_filtering_3D():
    # Create mock input data
    BlockGroup_basic = np.zeros((5, 8, 8))
    block_groups_noisy = np.zeros((5, 8, 8))
    sigma = 10
    mode = "cos"

    # Ensure the function returns a tuple of two numpy arrays
    block_groups_noisy, wiener_weight = filtering_3D(
        BlockGroup_basic, block_groups_noisy, sigma, mode
    )
    assert isinstance(block_groups_noisy, np.ndarray)
    assert isinstance(wiener_weight, float)

    # Ensure the shape of the returned arrays is as expected
    assert block_groups_noisy.shape == (5, 8, 8)


if __name__ == "__main__":
    pytest.main()
