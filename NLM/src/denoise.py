from multiprocessing import pool

import numpy as np
from numba import jit


def nlm_CEST(
    images,
    big_window_size,
    small_window_size,
    multi_processing: bool = False,
    pools: int = 5,
) -> np.ndarray:
    images = (images * 255).astype("int16")
    if not multi_processing:
        for dyn in range(images.shape[-1]):
            images[:, :, dyn] = nlm(
                images[:, :, dyn], big_window_size, small_window_size
            )
    else:
        with pool.Pool(pools) as p:
            res = [
                _
                for _ in p.imap_unordered(
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


def run_ml(args):
    img, big_window_size, small_window_size, dyn = args
    return nlm(img, big_window_size, small_window_size), dyn


#@jit(nopython=True)
def nlm(image, big_window_size, small_window_size):
    # Ensure that both window sizes are odd numbers
    if big_window_size % 2 == 0:
        big_window_size += 1
    if small_window_size % 2 == 0:
        small_window_size += 1

    pad_width, search_width = big_window_size // 2, small_window_size // 2

    # create padded image
    padded_image = pad_image(image, big_window_size, pad_width)

    # For each pixel in the actual image, find an area around the pixel that needs to be compared
    for image_x in range(pad_width, pad_width + image.shape[1]):
        for image_y in range(pad_width, pad_width + image.shape[0]):
            org_img_x_pixel = image_x - pad_width
            org_img_y_pixel = image_y - pad_width
            new_pixel_value, norm_factor = 0, 0

            # comparison neighbourhood
            comp_nbhd = get_comparison_neighborhood(
                padded_image, image_x, image_y, search_width
            )

            for small_window_x in range(
                org_img_x_pixel,
                org_img_x_pixel + big_window_size - small_window_size + 1,
                1,
            ):
                for small_window_y in range(
                    org_img_y_pixel,
                    org_img_y_pixel + big_window_size - small_window_size + 1,
                    1,
                ):
                    # find the small box
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

            new_pixel_value /= norm_factor
            image[org_img_y_pixel, org_img_x_pixel] = new_pixel_value

    return image



@jit(nopython=True)
def pad_image(image, big_window_size, pad_width):
    # TODO: Interpolate zero-borders to improve performance
    padded_image = np.zeros(
        (image.shape[0] + big_window_size, image.shape[1] + big_window_size)
    )
    padded_image[
        pad_width : pad_width + image.shape[0], pad_width : pad_width + image.shape[1]
    ] = image
    return padded_image


@jit(nopython=True)
def get_comparison_neighborhood(padded_image, x, y, search_width):
    return padded_image[
        y - search_width : y + search_width + 1, x - search_width : x + search_width + 1
    ]


@jit(nopython=True)
def get_small_neighborhood(padded_image, x, y, small_window_size):
    return padded_image[y : y + small_window_size, x : x + small_window_size]
