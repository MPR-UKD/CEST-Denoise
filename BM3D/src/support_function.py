import numpy as np
from numba import jit
from scipy.fftpack import dct, idct, dst, idst


def initialization(Img: np.ndarray, BlockSize: int, Kaiser_Window_beta: float):
    """
    :param Img: numpy image with shape n x m
    :param BlockSize: integer
    :param Kaiser_Window_beta: float - The $\beta$ parameter of the Kaiser window provides a convenient continuous
        control over the fundamental window trade-off between side-lobe level and main-lobe width.
        Larger $\beta$ values give lower side-lobe levels, but at the price of a wider main lobe.

    :return: init_img (zero image with shape n x m),
             init_weights (zero image with shape BlockSize x BlockSize),
             init_kaiser_window (kaiser window with shape BlockSize x BlockSize)
    """
    init_img = np.zeros(Img.shape, dtype=float)
    init_weights = np.zeros(Img.shape, dtype=float)
    window = np.matrix(np.kaiser(BlockSize, Kaiser_Window_beta))
    init_kaiser_window = np.array(window.T * window)
    return init_img, init_weights, init_kaiser_window


def compute_distance_of_two_blocks(block_1: np.ndarray, block_2: np.ndarray) -> float:
    """
    Compute the distance of two arrays after discrete transformation
    """
    return np.linalg.norm(block_1 - block_2) ** 2 / (block_1.shape[0] ** 2)


@jit(nopython=True)
def find_search_window(
    Img: np.ndarray, RefPoint: tuple, BlockSize: int, WindowSize: int
):
    window_location = np.zeros((2, 2), dtype="int16")

    window_location[0, 0] = max(
        0, RefPoint[0] + int((BlockSize - WindowSize) / 2)
    )  # left-top x
    window_location[0, 1] = max(
        0, RefPoint[1] + int((BlockSize - WindowSize) / 2)
    )  # left-top y
    window_location[1, 0] = window_location[0, 0] + WindowSize  # right-bottom x
    window_location[1, 1] = window_location[0, 1] + WindowSize  # right-bottom y

    if window_location[1, 0] >= Img.shape[0]:
        window_location[1, 0] = Img.shape[0] - 1
        window_location[0, 0] = window_location[1, 0] - WindowSize
    if window_location[1, 1] >= Img.shape[1]:
        window_location[1, 1] = Img.shape[1] - 1
        window_location[0, 1] = window_location[1, 1] - WindowSize

    return window_location


def discrete_2D_transformation(
    image: np.ndarray, BlockSize: int, mode: str
) -> np.ndarray:
    """
    Do discrete cosine / sinus transform (2D transform) for each block in image to reduce the complexity of
    applying transforms
    """

    # Init discrete_blocks
    discrete_blocks = np.zeros(
        (image.shape[0] - BlockSize, image.shape[1] - BlockSize, BlockSize, BlockSize),
        dtype=np.float64,
    )
    # Loop over image and transform each block
    for x in range(discrete_blocks.shape[0]):
        for y in range(discrete_blocks.shape[1]):
            block = image[x : x + BlockSize, y : y + BlockSize]

            if mode == "cos":
                discrete_blocks[x, y, :, :] = dct2D(block.astype(np.float64))
            elif mode == "sin":
                discrete_blocks[x, y, :, :] = dst2D(block.astype(np.float64))
    return discrete_blocks


def dct2D(array: np.ndarray) -> np.ndarray:
    """
    2D discrete cosine transform (DCT)
    """
    return dct(dct(array, axis=0, norm="ortho"), axis=1, norm="ortho")


def idct2D(array: np.ndarray) -> np.ndarray:
    """
    inverse 2D discrete cosine transform
    """
    return idct(idct(array, axis=0, norm="ortho"), axis=1, norm="ortho")


def dst2D(array: np.ndarray) -> np.ndarray:
    """
    2D discrete sinus transform (DCT)
    """
    return dst(dst(array, axis=0, norm="ortho"), axis=1, norm="ortho")


def idst2D(array: np.ndarray) -> np.ndarray:
    """
    inverse 2D discrete sinus transform
    """
    return idst(idst(array, axis=0, norm="ortho"), axis=1, norm="ortho")
