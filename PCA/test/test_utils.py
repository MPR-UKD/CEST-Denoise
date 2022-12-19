import unittest

import numpy as np
from PCA.src.utils import img_to_casorati_matrix, calc_eig


class TestUtils(unittest.TestCase):
    def test_casorati_matrix_mask_is_None(self):
        img = np.zeros((10, 10, 20))
        casorati = img_to_casorati_matrix(img)
        self.assertEqual(img.shape[0] * img.shape[1], casorati.shape[0])

    def test_casorati_matrix_with_mask(self):
        img = np.zeros((10, 10, 20))
        mask = np.zeros((10, 10))
        mask[:5, :5] = 1
        casorati = img_to_casorati_matrix(img, mask)
        self.assertEqual(mask.sum(), casorati.shape[0])

    def test_order_eigval(self):
        test_matrix = np.array([[0, 2], [2, 3]])
        eigvals, eigvecs = calc_eig(test_matrix, "max")
        self.assertEqual(eigvals.max(), eigvals[0])
        eigvals, eigvecs = calc_eig(test_matrix, "min")
        self.assertEqual(eigvals.min(), eigvals[0])


if __name__ == "__main__":
    unittest.main()
