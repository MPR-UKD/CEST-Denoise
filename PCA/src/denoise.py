import numpy as np
from .utils import *
from .criteria import *
import itertools
from typing import Tuple


def pca(
    img: np.ndarray, criteria: str | int, mask: np.ndarray | None = None
) -> np.ndarray:
    """
    :param img: noisy 2D CEST image (x,y,ndyn)
    :param mask: 2D binary mask (x,y)
    :return:
    """

    """ Step 1: Create column-wise mean-centered casorati_matrix C_tilde """
    C_tilde, Z_mean = step1(img, mask)
    """ Step 2: Principal component analysis - calc eigvals and eigvecs"""
    eigvals, eigvecs = step2(C_tilde)
    """ Step 3: Determine optimal number of components k """
    if type(criteria) == str:
        k = step3(eigvals, C_tilde, criteria)
    else:
        k = criteria
    print(k)
    """Step 4: Projection onto remaining components"""
    C_tilde = step4(C_tilde, Z_mean, eigvecs, k)
    """Step 5: Reform back to image"""
    img = step5(C_tilde, img, mask)
    return img


def step1(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    # Create casorati matrix and Subtract Z_mean to obtain column-wise mean-centered casorati_matrix C_tilde
    """
    casorati_matrix = img_to_casorati_matrix(img, mask)
    Z_mean = casorati_matrix.mean(axis=0)
    casorati_matrix -= Z_mean
    return casorati_matrix, Z_mean


def step2(C_tilde: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Step 2: Principal component analyses"""
    cov_C_tilde = cov(C_tilde)
    eigvals, eigvecs = calc_eig(cov_C_tilde, "max")
    return eigvals, eigvecs


def step3(eigenvalues: np.ndarray, C_tilde: np.ndarray, criteria: str) -> int:
    if criteria == "malinowski":
        return malinowski_criteria(eigenvalues, C_tilde.shape)
    if criteria == "nelson":
        return nelson_criteria(eigenvalues, C_tilde.shape)
    if criteria == "median":
        return median_criteria(eigenvalues)
    raise ValueError(
        f'Criteria: {criteria} is not implemented! Currently are "malinowski", "nelson" and "median"'
        f"criteria available"
    )


def step4(C_tilde, Z_mean, eigvecs, k) -> np.ndarray:
    C = C_tilde.copy()
    for i in range(C_tilde.shape[0]):
        C_tilde[i] = (
            np.array(
                [
                    np.dot(
                        C_tilde[i],
                        np.dot(
                            np.transpose(np.expand_dims(eigvecs[:, ii], axis=0)),
                            np.expand_dims(eigvecs[:, ii], axis=0),
                        ),
                    )
                    for ii in range(k)
                ]
            ).sum(axis=0)
            + Z_mean
        )
    return C_tilde


def step5(C_tilde, img, mask) -> np.ndarray:
    n, m, _ = img.shape
    if mask is None:
        mask = np.ones((n, m))
    count = 0
    for i1, i2 in itertools.product(range(n), range(m)):
        if mask[i2, i1] == 0:
            continue
        img[i2, i1, :] = C_tilde[count, :]
        count += 1
    return img
