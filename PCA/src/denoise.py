import itertools
from typing import Tuple, Optional, Union

import numpy as np

from .criteria import *
from .utils import *


def pca(img: np.ndarray, criteria: Union[str, int], mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Perform Principal Component Analysis (PCA) for denoising CEST images.

    Args:
        img (np.ndarray): A noisy 2D CEST image with dimensions (x, y, ndyn).
        criteria (Union[str, int]): A string representing the criteria used for selecting the number of
                                    principal components, or an integer representing the number of principal
                                    components to keep.
        mask (Optional[np.ndarray]): A 2D binary mask with dimensions (x, y). Defaults to None.

    Returns:
        np.ndarray: The denoised CEST image.
    """
    try:
        # Step 1: Create column-wise mean-centered casorati_matrix C_tilde
        C_tilde, Z_mean = step1(img, mask)

        # Step 2: Principal Component Analysis - calculate eigenvalues and eigenvectors
        eigvals, eigvecs = step2(C_tilde)

        # Step 3: Determine optimal number of components k
        k = criteria if type(criteria) == int else step3(eigvals, C_tilde, criteria)
        print(k)

        # Step 4: Projection onto remaining components
        C_tilde = step4(C_tilde, Z_mean, eigvecs, k)

        # Step 5: Reform back to image
        return step5(C_tilde, img, mask)
    except Exception as e:
        print(f"An error occurred: {e}")
        return img


def step1(img: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create casorati matrix and Subtract Z_mean to obtain column-wise mean-centered casorati_matrix C_tilde.

    Args:
        img (np.ndarray): The original noisy 2D CEST image.
        mask (Optional[np.ndarray]): A 2D binary mask. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The column-wise mean-centered casorati matrix and the mean of the casorati matrix.
    """
    casorati_matrix = img_to_casorati_matrix(img, mask)
    Z_mean = casorati_matrix.mean(axis=0)
    casorati_matrix -= Z_mean
    return casorati_matrix, Z_mean


def step2(C_tilde: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Principal Component Analysis on the given casorati matrix.

    Args:
        C_tilde (np.ndarray): The column-wise mean-centered casorati matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The eigenvalues and eigenvectors of the covariance matrix of C_tilde.
    """
    cov_C_tilde = cov(C_tilde)
    eigvals, eigvecs = calc_eig(cov_C_tilde, "max")
    return eigvals, eigvecs


def step3(eigvals: np.ndarray, C_tilde: np.ndarray, criteria: str) -> int:
    """
    Determine the optimal number of components based on the given criteria.

    Args:
        eigvals (np.ndarray): The eigenvalues obtained from PCA.
        C_tilde (np.ndarray): The column-wise mean-centered casorati matrix.
        criteria (str): The criteria used to select the number of principal components.

    Returns:
        int: The optimal number of principal components.

    Raises:
        ValueError: If an invalid criteria is provided.
    """
    if criteria == "malinowski":
        return malinowski_criteria(eigvals, C_tilde.shape)
    elif criteria == "nelson":
        return nelson_criteria(eigvals, C_tilde.shape)
    elif criteria == "median":
        return median_criteria(eigvals)
    else:
        raise ValueError(f'Invalid criteria: {criteria}. Supported criteria are "malinowski", "nelson", and "median".')


def step4(C_tilde: np.ndarray, Z_mean: np.ndarray, eigvecs: np.ndarray, k: int) -> np.ndarray:
    """
    Project the casorati matrix onto the remaining components.

    Args:
        C_tilde (np.ndarray): The column-wise mean-centered casorati matrix.
        Z_mean (np.ndarray): The mean of the casorati matrix.
        eigvecs (np.ndarray): The eigenvectors obtained from PCA.
        k (int): The number of principal components to keep.

    Returns:
        np.ndarray: The projected casorati matrix.
    """
    for i in range(C_tilde.shape[0]):
        C_tilde[i] = np.sum(
            [
                np.dot(
                    C_tilde[i],
                    np.dot(eigvecs[:, ii].reshape(-1, 1), eigvecs[:, ii].reshape(1, -1))
                )
                for ii in range(k)
            ],
            axis=0
        ) + Z_mean
    return C_tilde


def step5(C_tilde: np.ndarray, img: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """
    Convert the casorati matrix back to the 2D image format.

    Args:
        C_tilde (np.ndarray): The projected casorati matrix.
        img (np.ndarray): The original noisy 2D CEST image.
        mask (Optional[np.ndarray]): A 2D binary mask. Defaults to None.

    Returns:
        np.ndarray: The denoised 2D CEST image.
    """
    n, m, _ = img.shape
    mask = np.ones((n, m)) if mask is None else mask
    count = 0
    for i1, i2 in itertools.product(range(n), range(m)):
        if mask[i2, i1] == 0:
            continue
        img[i2, i1, :] = C_tilde[count, :]
        count += 1
    return img
