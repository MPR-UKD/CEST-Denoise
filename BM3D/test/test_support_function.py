import pytest
import numpy as np
from BM3D.src.support_function import *


# The test function takes in a function to test and the expected output
def test_initialization():
    Img = np.random.rand(10, 10)
    BlockSize = 3
    Kaiser_Window_beta = 2.0
    # Invoke the function being tested
    init_img, init_weights, init_kaiser_window = initialization(Img, BlockSize, Kaiser_Window_beta)

    # Assert that the output has the correct shape
    assert init_img.shape == Img.shape, 'Incorrect shape for init_img'
    assert init_weights.shape == Img.shape, 'Incorrect shape for init_weights'
    assert init_kaiser_window.shape == (BlockSize, BlockSize), 'Incorrect shape for init_kaiser_window'

    # Assert that the output is correct
    assert np.all(init_img == 0), 'init_img is not all zeros'
    assert np.all(init_weights == 0), 'init_weights is not all zeros'


# The test function takes in a function to test and the expected output
def helper_function_compute_distance_of_two_blocks(block_1, block_2, expected_distance):
    # Invoke the function being tested
    distance = compute_distance_of_two_blocks(block_1, block_2)

    # Assert that the output is correct
    assert distance == expected_distance, f'Expected distance {expected_distance}, got {distance}'

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
        helper_function_compute_distance_of_two_blocks(block_1, block_2, expected_distance)


if __name__ == '__main__':
    pytest.main()
