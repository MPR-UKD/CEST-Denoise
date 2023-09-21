from typing import Tuple

import numpy as np
from numba import jit
from scipy.fftpack import dct, idct, dst, idst


def initialization(
    Img: np.ndarray, BlockSize: int, Kaiser_Window_beta: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize images and generate the Kaiser window for BM3D filtering.

    Args:
        Img (np.ndarray): Input image of shape [n, m].
        BlockSize (int): Size of the block for processing.
        Kaiser_Window_beta (float): The beta parameter of the Kaiser window that controls the trade-off
            between side-lobe level and main-lobe width.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Zero image with shape [n, m]
            - Zero image with shape [BlockSize, BlockSize]
            - Kaiser window with shape [BlockSize, BlockSize]
    """
    # create zero image with shape n x m
    init_img = np.zeros(Img.shape, dtype=float)
    # create zero image with shape BlockSize x BlockSize
    init_weights = np.zeros(Img.shape, dtype=float)

    # generate kaiser window with specified beta value
    window = np.matrix(np.kaiser(BlockSize, Kaiser_Window_beta))
    # create 2D kaiser window by multiplying the window by its transpose
    init_kaiser_window = np.array(window.T * window)
    return init_img, init_weights, init_kaiser_window


def compute_distance_of_two_blocks(block_1: np.ndarray, block_2: np.ndarray) -> float:
    """
    Compute the squared Euclidean distance between two blocks.

    Args:
        block_1 (np.ndarray): First block.
        block_2 (np.ndarray): Second block.

    Returns:
        float: Distance between the two blocks.
    """
    # calculate the norm of the difference between the two blocks, squared, and divide by the square of the shape of
    # the blocks
    distance = np.linalg.norm(block_1 - block_2) ** 2 / (block_1.shape[0] ** 2)
    return distance


@jit(nopython=True)
def find_search_window(
    Img: np.ndarray, RefPoint: Tuple[int, int], BlockSize: int, WindowSize: int
) -> np.ndarray:
    """
    Determine the coordinates of the search window given a reference point.

    Args:
        Img (np.ndarray): Input image.
        RefPoint (Tuple[int, int]): Reference point coordinates.
        BlockSize (int): Size of the block for processing.
        WindowSize (int): Desired size of the search window.

    Returns:
        np.ndarray: Coordinates of the search window.
    """
    # Initialize an array to store the top-left and bottom-right coordinates of the search window
    window_location = np.zeros((2, 2), dtype="int16")

    # Calculate the top-left x coordinate of the search window
    # Start by taking the x coordinate of the reference point and adjusting it by (BlockSize - WindowSize) / 2
    # But make sure that the resulting value is at least 0 (to ensure it fits within the image)
    window_location[0, 0] = max(0, RefPoint[0] + int((BlockSize - WindowSize) / 2))

    # Calculate the top-left y coordinate of the search window using the same approach as above
    window_location[0, 1] = max(0, RefPoint[1] + int((BlockSize - WindowSize) / 2))

    # Calculate the bottom-right x coordinate by adding the WindowSize to the top-left x coordinate
    window_location[1, 0] = window_location[0, 0] + WindowSize

    # Calculate the bottom-right y coordinate by adding the WindowSize to the top-left y coordinate
    window_location[1, 1] = window_location[0, 1] + WindowSize

    # Check if the search window extends beyond the right edge of the image
    # If it does, set the bottom-right x coordinate to the right edge of the image
    # and adjust the top-left x coordinate accordingly
    if window_location[1, 0] >= Img.shape[0]:
        window_location[1, 0] = Img.shape[0] - 1
        window_location[0, 0] = window_location[1, 0] - WindowSize

    # Check if the search window extends beyond the bottom edge of the image
    # If it does, set the bottom-right y coordinate to the bottom edge of the image
    # and adjust the top-left y coordinate accordingly
    if window_location[1, 1] >= Img.shape[1]:
        window_location[1, 1] = Img.shape[1] - 1
        window_location[0, 1] = window_location[1, 1] - WindowSize

    # Return the top-left and bottom-right coordinates of the search window
    return window_location


def discrete_2D_transformation(
    image: np.ndarray, BlockSize: int, mode: str
) -> np.ndarray:
    """
    Perform a 2D discrete transformation (either cosine or sine) on each block of an image.

    Args:
        image (np.ndarray): Input image.
        BlockSize (int): Size of the block for processing.
        mode (str): Transformation mode, either "cos" for cosine or "sin" for sine.

    Returns:
        np.ndarray: Transformed blocks.
    """

    # Initialize a 4D array to store the transformed blocks
    # The first two dimensions correspond to the x and y position of each block in the image
    # The last two dimensions correspond to the size of each block (BlockSize x BlockSize)
    discrete_blocks = np.zeros(
        (image.shape[0] - BlockSize, image.shape[1] - BlockSize, BlockSize, BlockSize),
        dtype=np.float64,
    )

    # Loop over the image and transform each block
    for x in range(discrete_blocks.shape[0]):
        for y in range(discrete_blocks.shape[1]):
            # Extract the current block from the image
            block = image[x : x + BlockSize, y : y + BlockSize]

            # Apply either the discrete cosine transform (DCT) or discrete sine transform (DST) to the block
            # depending on the mode specified
            if mode == "cos":
                discrete_blocks[x, y, :, :] = dct2D(block.astype(np.float64))
            elif mode == "sin":
                discrete_blocks[x, y, :, :] = dst2D(block.astype(np.float64))
            else:
                raise ValueError

    # Return the transformed blocks
    return discrete_blocks


def dct2D(array: np.ndarray) -> np.ndarray:
    """
    Compute the 2D Discrete Cosine Transform of an array.

    Args:
        array (np.ndarray): Input 2D array.

    Returns:
        np.ndarray: Array after 2D DCT.
    """
    return dct(dct(array, axis=0, norm="ortho"), axis=1, norm="ortho")


def idct2D(array: np.ndarray) -> np.ndarray:
    """
    Compute the inverse 2D Discrete Cosine Transform of an array.

    Args:
        array (np.ndarray): Input 2D array.

    Returns:
        np.ndarray: Array after inverse 2D DCT.
    """
    return idct(idct(array, axis=0, norm="ortho"), axis=1, norm="ortho")


def dst2D(array: np.ndarray) -> np.ndarray:
    """
    Compute the 2D Discrete Sine Transform of an array.

    Args:
        array (np.ndarray): Input 2D array.

    Returns:
        np.ndarray: Array after 2D DST.
    """
    return dst(dst(array, axis=0, norm="ortho"), axis=1, norm="ortho")


def idst2D(array: np.ndarray) -> np.ndarray:
    """
    Compute the inverse 2D Discrete Sine Transform of an array.

    Args:
        array (np.ndarray): Input 2D array.

    Returns:
        np.ndarray: Array after inverse 2D DST.
    """
    return idst(idst(array, axis=0, norm="ortho"), axis=1, norm="ortho")
