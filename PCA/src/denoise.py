import numpy as np
from .utils import *


def pca(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    :param img: noisy 2D CEST image (x,y,ndyn)
    :param mask: 2D binary mask (x,y)
    :return:
    """

    """ Step 1: Create column-wise mean-centered casorati_matrix C_tilde """
    C_tilde = step1(img, mask)


def step1(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    # Create casorati matrix and Subtract Z_mean to obtain column-wise mean-centered casorati_matrix C_tilde
    """
    casorati_matrix = img_to_casorati_matrix(img, mask)
    Z_mean = casorati_matrix.mean(axis=0)
    casorati_matrix -= Z_mean
    return casorati_matrix
