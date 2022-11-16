import numpy as np


def pca(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    :param img: noisy 2D CEST image (x,y,ndyn)
    :param mask: 2D binary mask (x,y)
    :return:
    """

    """Step 1: Create Casorati matrix with size n ** 2 x ndyn"""
