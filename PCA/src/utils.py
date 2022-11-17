import numpy as np
from itertools import product


def img_to_casorati_matrix(img: np.array, mask: np.array = None) -> np.array:
    n, m, _ = img.shape
    if mask is None:
        mask = np.ones((n, m))
    casorati_matrix = []
    for i1, i2 in product(range(n), range(m)):
        if mask[i2, i1] == 0:
            continue
        casorati_matrix.append(img[i2, i1, :])
    return np.array(casorati_matrix)


def cov(C_tilde):
    # cov(C) = 1/ (n - 1) np.transpose(C_tilde) C_tilde = np.transpose(phi) ∧ phi
    # where:    n       - number of z-spectra dynamics
    #           C_tilde - colum-wise mean centered Casorati matrix
    #           phi     - n x n orthogonal eigenvector matrix
    #           ∧       - diag(λ_1, λ_2, ....) with λ_i being the eigenvalue i (λ_1 > λ_2 > .....)
    cov = 1 / (C_tilde.shape[1] - 1) * np.dot(np.transpose(C_tilde), C_tilde)
    return cov

def calc_eig(matrix: np.array, order: str) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eig(matrix)
    if order == 'max':
        idx = eigvals.argsort()[::-1]
    if order == 'min':
        idx = eigvals.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return np.real(eigvals), np.real(eigvecs)

