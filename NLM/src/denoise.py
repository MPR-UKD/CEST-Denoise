import numpy as np
from numba import jit


def nlm_CEST(images, big_window_size, small_window_size, multi_processing: bool) -> np.ndarray:
    if not multi_processing:
        for dyn in images.shape[-1]:
            images[:, :, dyn] = nlm(images[:, :, dyn], big_window_size, small_window_size)
    return images

@jit(nopython=True)
def nlm(image, big_window_size, small_window_size):
    pad_width, search_width = big_window_size // 2, small_window_size // 2

    # create padded image
    # Hint: The border consists only of zeros, so the ROI should not be on the border.
    # TODO: Interpolate zero-borders to improve performance
    paddedImage = np.zeros((image.shape[0] + big_window_size, image.shape[1] + big_window_size))
    paddedImage[pad_width:pad_width + image.shape[0], pad_width:pad_width + image.shape[1]] = image

    # For each pixel in the actual image, find a area around the pixel that needs to be compared
    for imageX in range(pad_width, pad_width + image.shape[1]):
        for imageY in range(pad_width, pad_width + image.shape[0]):

            org_img_x_pixel = imageX - pad_width
            org_img_y_pixel = imageY - pad_width
            new_pixel_value, norm_factor = 0, 0

            # comparison neighbourhood
            comp_nbhd = paddedImage[
                        imageY - search_width:imageY + search_width + 1,
                        imageX - search_width:imageX + search_width + 1]

            for small_window_x in range(org_img_x_pixel, org_img_x_pixel + big_window_size - small_window_size, 1):
                for small_window_y in range(org_img_y_pixel, org_img_y_pixel + big_window_size - small_window_size, 1):
                    # find the small box
                    small_nbhd = paddedImage[small_window_y:small_window_y + small_window_size + 1,
                                             small_window_x:small_window_x + small_window_size + 1]
                    euclidean_distance = np.linalg.norm(small_nbhd - comp_nbhd)
                    weight = np.exp(-euclidean_distance)
                    norm_factor += weight
                    new_pixel_value += weight * paddedImage[
                        small_window_y + search_width, small_window_x + search_width]

            new_pixel_value /= norm_factor
            image[org_img_y_pixel, org_img_x_pixel] = new_pixel_value

    return image
