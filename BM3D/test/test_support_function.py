import pytest
import numpy as np
from BM3D.src.support_function import *


# The test function takes in a function to test and the expected output
def test_initialization():
    Img = np.random.rand(10, 10)
    BlockSize = 3
    Kaiser_Window_beta = 2.0
    # Invoke the function being tested
    init_img, init_weights, init_kaiser_window = initialization(
        Img, BlockSize, Kaiser_Window_beta
    )

    # Assert that the output has the correct shape
    assert init_img.shape == Img.shape, "Incorrect shape for init_img"
    assert init_weights.shape == Img.shape, "Incorrect shape for init_weights"
    assert init_kaiser_window.shape == (
        BlockSize,
        BlockSize,
    ), "Incorrect shape for init_kaiser_window"

    # Assert that the output is correct
    assert np.all(init_img == 0), "init_img is not all zeros"
    assert np.all(init_weights == 0), "init_weights is not all zeros"


# The test function takes in a function to test and the expected output
def helper_function_compute_distance_of_two_blocks(block_1, block_2, expected_distance):
    # Invoke the function being tested
    distance = compute_distance_of_two_blocks(block_1, block_2)

    # Assert that the output is correct
    assert (
        distance == expected_distance
    ), f"Expected distance {expected_distance}, got {distance}"


# Test the function with some input data
def test_compute_distance_of_two_blocks_inputs():
    block_1 = np.array([[1, 2], [3, 4]])
    block_2 = np.array([[5, 6], [7, 8]])
    expected_distance = 16.0
    helper_function_compute_distance_of_two_blocks(block_1, block_2, expected_distance)


# Test the function with identical input blocks
def test_compute_distance_of_two_blocks_identical():
    block_1 = np.array([[1, 2], [3, 4]])
    block_2 = np.array([[1, 2], [3, 4]])
    expected_distance = 0.0
    helper_function_compute_distance_of_two_blocks(block_1, block_2, expected_distance)


# Test the function with input blocks of different shapes
def test_compute_distance_of_two_blocks_different_shapes():
    block_1 = np.array([[1, 2], [3, 4]])
    block_2 = np.array([[5, 6], [7, 8], [9, 10]])
    expected_distance = None
    with pytest.raises(ValueError):
        helper_function_compute_distance_of_two_blocks(
            block_1, block_2, expected_distance
        )


# The test function takes in a function to test and the expected output
def helper_function_find_search_window(
    Img, RefPoint, BlockSize, WindowSize, expected_window_location
):
    # Invoke the function being tested
    window_location = find_search_window(Img, RefPoint, BlockSize, WindowSize)

    # Assert that the output is correct
    assert np.array_equal(
        window_location, expected_window_location
    ), f"Expected window location {expected_window_location}, got {window_location}"


# Test the function with a reference point in the middle of the image
def test_find_search_window_middle():
    Img = np.zeros((10, 10))
    RefPoint = (5, 5)
    BlockSize = 3
    WindowSize = 3
    expected_window_location = np.array([[5, 5], [8, 8]])
    helper_function_find_search_window(
        Img, RefPoint, BlockSize, WindowSize, expected_window_location
    )


# Test the function with a reference point near the top left corner of the image
def test_find_search_window_top_left():
    Img = np.zeros((10, 10))
    RefPoint = (0, 0)
    BlockSize = 3
    WindowSize = 3
    expected_window_location = np.array([[0, 0], [3, 3]])
    helper_function_find_search_window(
        Img, RefPoint, BlockSize, WindowSize, expected_window_location
    )


# Test the function with a reference point near the bottom right corner of the image
def test_find_search_window_bottom_right():
    Img = np.zeros((10, 10))
    RefPoint = (9, 9)
    BlockSize = 3
    WindowSize = 3
    expected_window_location = np.array([[6, 6], [9, 9]])
    helper_function_find_search_window(
        Img, RefPoint, BlockSize, WindowSize, expected_window_location
    )


# Test the function with a WindowSize larger than the BlockSize
def test_find_search_window_large_window():
    Img = np.zeros((10, 10))
    RefPoint = (5, 5)
    BlockSize = 3
    WindowSize = 5
    expected_window_location = np.array([[4, 4], [9, 9]])
    helper_function_find_search_window(
        Img, RefPoint, BlockSize, WindowSize, expected_window_location
    )


# Ensure that the function is correct for a range of inputs
@pytest.mark.parametrize("BlockSize", [2, 4, 8])
@pytest.mark.parametrize("mode", ["cos", "sin"])
def test_discrete_2D_transformation(BlockSize, mode):
    # Create a test image
    image = np.random.rand(10, 10)

    # Compute the transformed blocks using the function under test
    transformed_blocks = discrete_2D_transformation(image, BlockSize, mode)

    # Check that the shape of the transformed blocks is correct
    assert transformed_blocks.shape == (
        image.shape[0] - BlockSize,
        image.shape[1] - BlockSize,
        BlockSize,
        BlockSize,
    )


# Ensure that the function raises an error when given invalid input
def test_discrete_2D_transformation_invalid_input():
    # Create a test image
    image = np.random.rand(10, 10, 10, 10)

    # Block size of 0 should raise an error
    with pytest.raises(ValueError):
        discrete_2D_transformation(image, 0, "cos")

    # Invalid mode should raise an error
    with pytest.raises(ValueError):
        discrete_2D_transformation(image, 2, "invalid")


if __name__ == "__main__":
    pytest.main()
