from multiprocessing import pool

import numpy as np

from .step1 import step1_basic_estimation
from .step2 import step2_final_estimation


def bm3d_CEST(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    config: dict | None = None,
    multi_processing: bool = False,
) -> np.ndarray:
    """
    Apply BM3D algorithm on an image using the provided configuration.

    Args:
        img (np.ndarray): The input image.
        mask (np.ndarray, optional): The mask applied on the image. Defaults to None.
        config (dict, optional): Configuration for the BM3D algorithm. Defaults to None.
        multi_processing (bool, optional): If True, use multiprocessing for enhanced performance. Defaults to False.

    Returns:
        np.ndarray: The denoised image.
    """
    img = (img * 255).astype("int16")
    if not multi_processing:
        for dyn in range(img.shape[-1]):
            if len(img.shape) == 3:
                img[:, :, dyn] = bm3d(img[:, :, dyn], config, mask)
            elif len(img.shape) == 4:
                img[:, :, 0, dyn] = bm3d(img[:, :, 0, dyn], config, mask)
    else:
        with pool.Pool(42) as p:
            res = p.imap_unordered(
                run_ml,
                [(img[:, :, dyn], mask, config, dyn) for dyn in range(img.shape[-1])],
            )
            for d_img, dyn in res:
                img[:, :, dyn] = d_img
    return img / 255.0


def run_ml(args: tuple) -> tuple:
    """
    Helper function to unpack arguments for multiprocessing.

    Args:
        args (tuple): The arguments for the bm3d function.

    Returns:
        tuple: The denoised image and the dynamic index.
    """
    img, mask, config, dyn = args
    return bm3d(img, config, mask), dyn


def bm3d(
    img: np.ndarray, config: dict | None = None, mask: np.ndarray | None = None
) -> np.ndarray:
    """
    BM3D algorithm for denoising an image.

    Args:
        img (np.ndarray): The input image.
        config (dict, optional): Configuration for the BM3D algorithm. Defaults to None.
        mask (np.ndarray, optional): The mask applied on the image. Defaults to None.

    Returns:
        np.ndarray: The denoised image.
    """
    assert "int" in str(img.dtype), "Only integer images currently supported"

    if config is None:
        config = dict()

    def confirm_size(value: int) -> bool:
        """Check if the given value exceeds image dimensions."""
        return not (value > img.shape[0] or value > img.shape[1])

    # Configuration extraction
    sigma = config.get("sigma", 25.0)
    lamb2d = config.get("lamb2d", 2.0)
    lamb3d = config.get("lamb3d", 2.7)
    KaiserWindowBeta = config.get("KaiserWindowBeta", 2.0)
    transformation_function = config.get("TransformationFunction", "cos")

    assert transformation_function in ["cos", "sin"], (
        f"The transformation function {transformation_function} is not supported."
        f" Supported functions are: [cos, sin]"
    )

    if mask is None:
        mask = np.ones_like(img)

    # Step 1 configuration
    step1_threshold_distance = config.get("step1_threshold_distance", 2500)
    step1_max_match = config.get("step1_max_match", 16)
    step1_BlockSize = config.get("step1_BlockSize", 8)
    step1_WindowSize = config.get("step1_WindowSize", 39)

    if not confirm_size(step1_BlockSize) or not confirm_size(step1_WindowSize):
        raise ValueError("Step 1 configuration values exceed image dimensions.")

    # Step 2 configuration
    step2_threshold_distance = config.get("step2_threshold_distance", 400)
    step2_max_match = config.get("step2_max_match", 32)
    step2_BlockSize = config.get("step2_BlockSize", 8)
    step2_WindowSize = config.get("step2_WindowSize", 39)

    if not confirm_size(step2_BlockSize) or not confirm_size(step2_WindowSize):
        raise ValueError("Step 2 configuration values exceed image dimensions.")

    param_step1 = (
        step1_BlockSize,
        step1_threshold_distance,
        step1_max_match,
        step1_WindowSize,
        lamb2d,
        lamb3d,
        sigma,
        KaiserWindowBeta,
        transformation_function,
    )

    param_step2 = (
        step2_BlockSize,
        step2_threshold_distance,
        step2_max_match,
        step2_WindowSize,
        sigma,
        KaiserWindowBeta,
        transformation_function,
    )

    # Basic estimate
    basic_img = step1_basic_estimation(img, param_step1, mask)

    # Final estimate
    final_img = step2_final_estimation(basic_img, img, param_step2, mask)

    return final_img
