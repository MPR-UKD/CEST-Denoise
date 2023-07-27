import numpy as np
from itertools import product


def img_to_casorati_matrix(img: np.array, mask: np.array = None) -> np.array:
    # Get the shape of the image
    n, m, _ = img.shape

    # If a mask is not provided, create a default mask of all 1's
    if mask is None:
        mask = np.ones((n, m))

    # Initialize an empty list to store the pixel values
    casorati_matrix = []

    # Iterate through all pairs of indices (i1, i2) in the image
    for i1, i2 in product(range(n), range(m)):
        # If the mask value at this index is 0, skip this iteration
        if mask[i2, i1] == 0:
            continue

        # Otherwise, append the pixel value at this index to the list
        casorati_matrix.append(img[i2, i1, :])

    # Convert the list of pixel values to a numpy array and return it
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
    # Calculate the eigenvalues and eigenvectors of the matrix
    eigvals, eigvecs = np.linalg.eig(matrix)

    # Determine which eigenvalues to sort based on the specified order
    if order == "max":
        # Sort the eigenvalues in decreasing order
        idx = eigvals.argsort()[::-1]
    if order == "min":
        # Sort the eigenvalues in increasing order
        idx = eigvals.argsort()

    # Sort the eigenvalues and eigenvectors based on the indices obtained above
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Return the real parts of the sorted eigenvalues and eigenvectors
    return np.real(eigvals), np.real(eigvecs)
