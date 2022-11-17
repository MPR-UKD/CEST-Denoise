import numpy as np


def malinowski_criteria(eigvals: np.ndarray, C_tilde_shape: tuple):
    """Based on a theory of error concerning abstact factor analysis"""
    RE = []
    eigvals = np.sort(eigvals)[::-1]
    eigvals_liste = []
    for k in range(C_tilde_shape[1] - 1):
        l = k + 1
        eigvals_sum = np.sum(eigvals[l::])
        eigvals_liste.append((eigvals_sum))
        if C_tilde_shape[0] > C_tilde_shape[1]:
            RE.append((abs(eigvals_sum) / (C_tilde_shape[0] * (C_tilde_shape[1] - k))) ** 0.5)
        elif C_tilde_shape[0] < C_tilde_shape[1]:
            RE.append(abs((eigvals_sum / (C_tilde_shape[1] * (C_tilde_shape[0] - k)))) ** 0.5)

    k_ind = []
    for k in range(len(RE)):
        k_ind.append(RE[k] / ((C_tilde_shape[1] - k) ** 2))

    k_min = k_ind.index(min(k_ind))
    return k_min


def nelson_criteria(eigvals: np.ndarray, C_tilde_shape: tuple):
    """Based on the shape of the course of the eigenvalues plotted as a diminishing series"""
    r_list = []
    l_ist = np.arange(len(eigvals))
    for k in range(C_tilde_shape[1] - 1):
        l = k + 1
        eigvals_sum_l = np.sum(l_ist[l::] * eigvals[l::])
        eigvals_sum = np.sum(eigvals[l::])
        eigvals_sum_squared = np.sum(eigvals[l::] ** 2)
        l_sum = np.sum(l_ist[l::])
        l_squared = np.sum(l_ist[l::] ** 2)
        numerator = (C_tilde_shape[1] - k) * eigvals_sum_l - l_sum * eigvals_sum
        denominator = np.sqrt(((C_tilde_shape[1] - k) * l_squared - l_sum ** 2)) * np.sqrt(
            ((C_tilde_shape[1] - k) * eigvals_sum_squared - eigvals_sum ** 2))
        if denominator != 0:
            r_squared = (numerator / denominator) ** 2
            r_list.append(r_squared)
    r_list = np.array(r_list)

    for i in range(len(r_list) - 1):
        dif = r_list[i + 1] - r_list[i]
        if dif > 0:
            continue
        else:
            k_reg = np.where(r_list == r_list[i])[0]
            break
    return k_reg[0]


def median_criteria(eigvals: np.ndarray):
    """
    Approach uses the median of the eigencalues to estimate the noise lvel of the data and thereby determines
    a threshold for the signal related eigenvalues
    """
    median = np.median(eigvals)
    eigvals *= (1 / median)
    eigvals_t = []
    # first condition
    for i in range(len(eigvals)):
        if np.sqrt(eigvals[i]) < 2:
            eigvals_t.append(eigvals[i] * abs(median))
    eigvals_t = np.array(eigvals_t)

    # second conditions
    PC_list = []
    beta = 1.29
    median_t = np.median(eigvals_t)
    # print(median_t)
    # get principal components
    for i in range(len(eigvals_t)):
        if eigvals_t[i] >= median_t * beta ** 2:
            PC_list.append(eigvals_t[i])

    if len(PC_list) > 0:
        # k_med = PC_list.index(max(PC_list))
        k_med = np.where(eigvals == max(PC_list))[0]
    else:
        k_med = [eigvals.shape[0] - 1]

    return k_med[0]
