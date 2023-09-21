from itertools import product
from typing import Tuple, Optional

import numpy as np


def img_to_casorati_matrix(img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Converts a given 2D image to a Casorati matrix.

    Args:
        img (np.ndarray): The 2D image to be converted. Shape (n, m, _).
        mask (Optional[np.ndarray]): The binary mask applied to the image. Shape (n, m). Defaults to None.

    Returns:
        np.ndarray: The Casorati matrix obtained from the provided image.
    """
    n, m, _ = img.shape
    if mask is None:
        mask = np.ones((n, m))

    casorati_matrix = [img[i2, i1, :] for i1, i2 in product(range(n), range(m)) if mask[i2, i1] != 0]

    return np.array(casorati_matrix)


def cov(C_tilde: np.ndarray) -> np.ndarray:
    """
    Calculates the covariance matrix of the provided Casorati matrix.

    Args:
        C_tilde (np.ndarray): The column-wise mean centered Casorati matrix.

    Returns:
        np.ndarray: The covariance matrix of the provided Casorati matrix.
    """
    n = C_tilde.shape[1]
    return 1 / (n - 1) * np.dot(np.transpose(C_tilde), C_tilde)


def calc_eig(matrix: np.ndarray, order: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates and sorts the eigenvalues and eigenvectors of the provided matrix.

    Args:
        matrix (np.ndarray): The input matrix for which eigenvalues and eigenvectors are to be calculated.
        order (str): A string indicating the order to sort the eigenvalues.
                    'max' for descending and 'min' for ascending.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - A sorted array of eigenvalues.
            - A 2D array where each column is a sorted eigenvector.
    """
    eigvals, eigvecs = np.linalg.eig(matrix)

    if order == "max":
        idx = eigvals.argsort()[::-1]
    elif order == "min":
        idx = eigvals.argsort()
    else:
        raise ValueError("Order must be either 'max' or 'min'.")

    return np.real(eigvals[idx]), np.real(eigvecs[:, idx])
