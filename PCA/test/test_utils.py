import unittest

import numpy as np
from PCA.src.utils import img_to_casorati_matrix


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



if __name__ == '__main__':
    unittest.main()
