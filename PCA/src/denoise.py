import numpy as np
from .utils import *
from .criteria import *


def pca(img: np.ndarray, mask: np.ndarray, criteria: str) -> np.ndarray:
    """
    :param img: noisy 2D CEST image (x,y,ndyn)
    :param mask: 2D binary mask (x,y)
    :return:
    """

    """ Step 1: Create column-wise mean-centered casorati_matrix C_tilde """
    C_tilde = step1(img, mask)
    """ Step 2: Principal component analysis - calc eigvals and eigvecs"""
    eigvals, eigvecs = step2(C_tilde)
    """ Step 3: Determine optimal number of components k """
    k = step3(eigvals, C_tilde, criteria)


def step1(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    # Create casorati matrix and Subtract Z_mean to obtain column-wise mean-centered casorati_matrix C_tilde
    """
    casorati_matrix = img_to_casorati_matrix(img, mask)
    Z_mean = casorati_matrix.mean(axis=0)
    casorati_matrix -= Z_mean
    return casorati_matrix


def step2(C_tilde):
    """Step 2: Principal component analyses"""
    cov_C_tilde = np.cov(C_tilde)
    eigvals, eigvecs = calc_eig(cov_C_tilde, 'max')
    return eigvals, eigvecs


def step3(eigenvalues: np.ndarray, C_tilde: np.ndarray, criteria: str):
    if criteria == 'malinowski':
        return malinowski_criteria(eigenvalues, C_tilde.shape)
    if criteria == 'nelson':
        return nelson_criteria(eigenvalues, C_tilde.shape)
    if criteria == 'median':
        return median_criteria(eigenvalues)
    raise ValueError(f'Criteria: {criteria} is not implemented! Currently are "malinowski", "nelson" and "median"'
                     f'criteria available')
