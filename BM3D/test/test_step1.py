import pytest
import numpy as np

from BM3D.src.step1 import *
from BM3D.src.support_function import compute_distance_of_two_blocks


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


def test_grouping():
    # create test inputs
    noisy_image = np.random.rand(10, 10)
    ref_block_position = (5, 5)
    discrete_transform_blocks = np.random.rand(10, 10, 3, 3)
    block_size = 3
    threshold_distance = 0.5
    max_number_of_similar_blocks = 5
    window_size = 7

    # call grouping function
    block_positions, block_groups = grouping(
        noisy_image,
        ref_block_position,
        discrete_transform_blocks,
        block_size,
        threshold_distance,
        max_number_of_similar_blocks,
        window_size,
    )

    # check if the output has the correct shape
    assert block_positions.shape == (max_number_of_similar_blocks, 2)
    assert block_groups.shape == (max_number_of_similar_blocks, block_size, block_size)

    # check if the output blocks are indeed similar to the reference block
    reference_block = discrete_transform_blocks[
        ref_block_position[0], ref_block_position[1], :, :
    ]
    for i in range(block_groups.shape[0]):
        assert (
            compute_distance_of_two_blocks(reference_block, block_groups[i])
            < threshold_distance
        )

    reference_block = discrete_transform_blocks[
        ref_block_position[0], ref_block_position[1], :, :
    ]
    # check if the output blocks are sorted in increasing order of similarity
    for i in range(1, block_groups.shape[0]):
        assert compute_distance_of_two_blocks(
            block_groups[i - 1], reference_block
        ) < compute_distance_of_two_blocks(block_groups[i], reference_block)

    # create test inputs with more than max_number_of_similar_blocks similar blocks
    threshold_distance = 0.1
    max_number_of_similar_blocks = 1

    # call grouping function
    block_positions, block_groups = grouping(
        noisy_image,
        ref_block_position,
        discrete_transform_blocks,
        block_size,
        threshold_distance,
        max_number_of_similar_blocks,
        window_size,
    )

    # check if the number of output blocks is equal to max_number_of_similar_blocks
    assert block_groups.shape[0] == max_number_of_similar_blocks


def test_filtering_3d():
    # create test input
    block_group = np.random.rand(5, 3, 3) + 0.4
    threshold = 0.5
    mode = "cos"

    # call filtering_3d function
    filtered_block_group, count = filtering_3d(block_group, threshold, mode)

    # check if the output has the correct shape
    assert filtered_block_group.shape == (5, 3, 3)

    # check if the output count is correct
    count = 0
    for x, y in itertools.product(
        range(block_group.shape[1]), range(block_group.shape[2])
    ):
        # apply the discrete transform to the original block
        if mode == "cos":
            transformed_original = dct(block_group[:, x, y], norm="ortho")
        elif mode == "sin":
            transformed_original = dst(block_group[:, x, y], norm="ortho")
        else:
            # If the mode is not "cos" or "sin", raise an error
            raise NotImplementedError
        # update the count
        count += np.nonzero(
            np.where(abs(transformed_original) >= threshold, transformed_original, 0)
        )[0].size
    # check if the output count is equal to the calculated count
    # call filtering_3d function
    filtered_block_group, output_count = filtering_3d(block_group, threshold, mode)
    assert count == output_count


def test_aggregation():
    # create test inputs
    block_group = np.random.rand(3, 5, 5)
    block_positions = np.array([[0, 0], [5, 5], [10, 10]])
    basic_estimate_img = np.zeros((15, 15))
    basic_weight = np.zeros((15, 15))
    basicKaiser = np.ones((5, 5))
    nonzero_cnt = 3
    sigma = 1.0
    mode = "cos"

    # call aggregation function
    aggregation(
        block_group,
        block_positions,
        basic_estimate_img,
        basic_weight,
        basicKaiser,
        nonzero_cnt,
        sigma,
        mode,
    )

    # check if the output basic estimate image has the correct shape
    assert basic_estimate_img.shape == (15, 15)
    # check if the output basic weight image has the correct shape
    assert basic_weight.shape == (15, 15)


if __name__ == "__main__":
    pytest.main()
