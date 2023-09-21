from multiprocessing import Pool

import numpy as np
from numba import jit


def nlm_CEST(
    images: np.ndarray,
    big_window_size: int,
    small_window_size: int,
    multi_processing: bool = False,
    pools: int = 5,
) -> np.ndarray:
    """
    Perform Non-Local Means (NLM) denoising on CEST images.

    Args:
        images (np.ndarray): 3D numpy array representing the set of images to be denoised.
        big_window_size (int): Size of the large search window.
        small_window_size (int): Size of the small comparison window.
        multi_processing (bool, optional): Whether to use multiprocessing. Defaults to False.
        pools (int, optional): Number of worker processes to use if multi_processing is True. Defaults to 5.

    Returns:
        np.ndarray: The denoised images.
    """
    # Normalize image intensities to 8-bit (0-255) range and convert to integer type
    images = (images * 255).astype("int16")

    if not multi_processing:
        for dyn in range(images.shape[-1]):
            images[:, :, dyn] = nlm(
                images[:, :, dyn], big_window_size, small_window_size
            )
    else:
        with Pool(pools) as p:
            res = [
                result
                for result in p.imap_unordered(
                    run_ml,
                    [
                        (images[:, :, dyn], big_window_size, small_window_size, dyn)
                        for dyn in range(images.shape[-1])
                    ],
                )
            ]
        for d_img, dyn in res:
            images[:, :, dyn] = d_img

    return images / 255


def run_ml(args: tuple) -> tuple:
    """
    Unpacks arguments and runs the NLM denoising function for multiprocessing.

    Args:
        args (tuple): A tuple containing the image, big window size, small window size, and dyn.

    Returns:
        tuple: The denoised image and dyn.
    """
    img, big_window_size, small_window_size, dyn = args
    return nlm(img, big_window_size, small_window_size), dyn


@jit(nopython=True)
def nlm(image: np.ndarray, big_window_size: int, small_window_size: int) -> np.ndarray:
    """
    Perform Non-Local Means (NLM) denoising on a single 2D image.

    Args:
        image (np.ndarray): 2D numpy array representing the image to be denoised.
        big_window_size (int): Size of the large search window.
        small_window_size (int): Size of the small comparison window.

    Returns:
        np.ndarray: The denoised image.
    """
    # Ensure that both window sizes are odd numbers
    if big_window_size % 2 == 0:
        big_window_size += 1
    if small_window_size % 2 == 0:
        small_window_size += 1

    pad_width = big_window_size // 2
    search_width = small_window_size // 2

    padded_image = pad_image(image, big_window_size, pad_width)

    # Denoising each pixel in the original image
    for image_x in range(pad_width, pad_width + image.shape[1]):
        for image_y in range(pad_width, pad_width + image.shape[0]):
            org_img_x_pixel = image_x - pad_width
            org_img_y_pixel = image_y - pad_width
            new_pixel_value, norm_factor = 0, 0

            comp_nbhd = get_comparison_neighborhood(
                padded_image, image_x, image_y, search_width
            )

            for small_window_x in range(
                org_img_x_pixel,
                org_img_x_pixel + big_window_size - small_window_size + 1,
            ):
                for small_window_y in range(
                    org_img_y_pixel,
                    org_img_y_pixel + big_window_size - small_window_size + 1,
                ):
                    small_nbhd = get_small_neighborhood(
                        padded_image, small_window_x, small_window_y, small_window_size
                    )
                    euclidean_distance = np.linalg.norm(small_nbhd - comp_nbhd)
                    weight = np.exp(-euclidean_distance)
                    norm_factor += weight
                    new_pixel_value += (
                        weight
                        * padded_image[
                            small_window_y + search_width, small_window_x + search_width
                        ]
                    )

            image[org_img_y_pixel, org_img_x_pixel] = new_pixel_value / norm_factor

    return image


@jit(nopython=True)
def pad_image(image: np.ndarray, big_window_size: int, pad_width: int) -> np.ndarray:
    """
    Pad the given image with zeros around its borders.

    Args:
        image (np.ndarray): 2D numpy array representing the original image.
        big_window_size (int): Size of the large search window.
        pad_width (int): The width of the padding around the image.

    Returns:
        np.ndarray: The padded image.
    """
    # Creating a new zeroed (padded) image
    padded_image = np.zeros(
        (image.shape[0] + big_window_size, image.shape[1] + big_window_size)
    )
    padded_image[
        pad_width : pad_width + image.shape[0], pad_width : pad_width + image.shape[1]
    ] = image
    return padded_image


@jit(nopython=True)
def get_comparison_neighborhood(
    padded_image: np.ndarray, x: int, y: int, search_width: int
) -> np.ndarray:
    """
    Extract the comparison neighborhood from the padded image.

    Args:
        padded_image (np.ndarray): 2D numpy array representing the padded image.
        x (int): The x-coordinate of the center pixel of the comparison neighborhood.
        y (int): The y-coordinate of the center pixel of the comparison neighborhood.
        search_width (int): The width of the search area around the pixel.

    Returns:
        np.ndarray: The comparison neighborhood.
    """
    return padded_image[
        y - search_width : y + search_width + 1, x - search_width : x + search_width + 1
    ]


@jit(nopython=True)
def get_small_neighborhood(
    padded_image: np.ndarray, x: int, y: int, small_window_size: int
) -> np.ndarray:
    """
    Extract a small neighborhood from the padded image.

    Args:
        padded_image (np.ndarray): 2D numpy array representing the padded image.
        x (int): The x-coordinate of the top-left corner of the small neighborhood.
        y (int): The y-coordinate of the top-left corner of the small neighborhood.
        small_window_size (int): The size of the small neighborhood.

    Returns:
        np.ndarray: The extracted small neighborhood.
    """
    return padded_image[y : y + small_window_size, x : x + small_window_size]
