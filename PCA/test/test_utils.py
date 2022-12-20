import pytest

from PCA.src.utils import *


def test_img_to_casorati_matrix():
    # Create a sample image and mask
    img = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    mask = np.array([[1, 0], [1, 1]])

    # Call the function with the sample image and mask
    casorati_matrix = img_to_casorati_matrix(img, mask)

    # Verify that the output is correct
    expected_output = np.array([[1, 2, 3], [7, 8, 9], [10, 11, 12]])
    assert (casorati_matrix == expected_output).all()


def test_calc_eig():
    # Create a sample matrix
    matrix = np.array([[1, 2], [3, 4]])

    # Call the function with the sample matrix and the "max" order
    eigvals, eigvecs = calc_eig(matrix, "max")

    expected_eigvals = np.array([(33 ** (1 / 2) + 5) / 2, (-(33 ** (1 / 2)) + 5) / 2])
    assert (eigvals == expected_eigvals).all()

    # Call the function with the sample matrix and the "min" order
    eigvals, eigvecs = calc_eig(matrix, "min")

    # Verify that the output is correct
    expected_eigvals = np.array([(-(33 ** (1 / 2)) + 5) / 2, (33 ** (1 / 2) + 5) / 2])
    assert (eigvals == expected_eigvals).all()


if __name__ == "__main__":
    pytest.main()
