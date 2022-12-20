import numpy as np

import numpy as np


def malinowski_criteria(eigenvalues, C_tilde_shape):
    """
    Determines the optimal number of k for the Malinowski method for adaptive noise reduction in CEST MRI data.

    Parameters:
    - C_tilde: the noise covariance matrix
    - eigenvalues: the sorted eigenvalues of C_tilde

    Returns:
    - k_opt: the optimal number of k
    """

    m, n = C_tilde_shape[0], C_tilde_shape[1]
    #
    eigenvalues = abs(eigenvalues)
    # eigvals /= eigvals.min()
    m = 1
    RE = np.zeros(n - 1)
    for k in range(n - 1):
        RE[k] = np.sqrt(np.sum(eigenvalues[k + 1 :]) / (m * (n - k)))
    k_opt = np.argmin(np.array([RE[k] / (m * (n - k) ** 2) for k in range(n - 1)]))
    return k_opt


def nelson_criteria(eigvals: np.ndarray, C_tilde_shape: tuple):
    """Based on the shape of the course of the eigenvalues plotted as a diminishing series"""
    r2_list = []
    l_ist = np.arange(len(eigvals))
    for k in range(C_tilde_shape[1] - 1):
        l = k + 1
        eigvals_sum_l = np.sum(l_ist[l::] * eigvals[l::])
        eigvals_sum = np.sum(eigvals[l::])
        eigvals_sum_squared = np.sum(eigvals[l::] ** 2)
        l_sum = np.sum(l_ist[l::])
        l_squared = np.sum(l_ist[l::] ** 2)
        numerator = (C_tilde_shape[1] - k) * eigvals_sum_l - l_sum * eigvals_sum
        denominator = np.sqrt(
            ((C_tilde_shape[1] - k) * l_squared - l_sum**2)
        ) * np.sqrt(((C_tilde_shape[1] - k) * eigvals_sum_squared - eigvals_sum**2))
        if denominator != 0:
            r_squared = (numerator / denominator) ** 2
            r2_list.append(r_squared)

    for i in range(len(r2_list) - 1):
        dif = r2_list[i + 1] - r2_list[i]
        if dif > 0 or r2_list[i] < 0.8:
            continue
        else:
            return i
    return len(r2_list)


def median_criteria(eigvals: np.ndarray):
    """
    Approach uses the median of the eigencalues to estimate the noise lvel of the data and thereby determines
    a threshold for the signal related eigenvalues
    """
    median = np.median(eigvals)
    eigvals_t = eigvals[eigvals < 2 * median]
    beta2 = 1.29**2
    k_med = eigvals[eigvals > beta2 * np.median(eigvals_t)].shape[0]
    return k_med
