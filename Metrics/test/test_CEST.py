import unittest
from Metrics.src.CEST import mtr_asym, mtr_asym_curve
import numpy as np
from test_support_function.src.CEST import generate_Z_3D


class TestMtrAsym(unittest.TestCase):
    def test_mtr_asym_curve(self):
        # Test MTR asymmetry curve with a Lorentzian curve centered at 0 ppm
        Z = generate_Z_3D(img_size=(2, 2), dyn=5, ppm=3, a=1, b=0, c=3)
        expected_result = np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
        result = mtr_asym_curve(Z)
        np.testing.assert_array_equal(result, expected_result)

        # Test MTR asymmetry curve with a Lorentzian curve centered at positive ppm
        Z = generate_Z_3D(img_size=(2, 2), dyn=5, ppm=3, a=1, b=1, c=3)
        result = mtr_asym_curve(Z)
        self.assertTrue(result.sum() > 0)

        # Test MTR asymmetry curve with a Lorentzian curve centered at negative ppm
        Z = generate_Z_3D(img_size=(2, 2), dyn=5, ppm=3, a=1, b=-1, c=3)
        result = mtr_asym_curve(Z)
        self.assertTrue(result.sum() < 0)

        # Test MTR asymmetry curve with a Lorentzian curve with a lower height
        Z1 = generate_Z_3D(img_size=(2, 2), dyn=5, ppm=3, a=1, b=1, c=3)
        result1 = mtr_asym_curve(Z1)
        Z2 = generate_Z_3D(img_size=(2, 2), dyn=5, ppm=3, a=0.5, b=1, c=3)
        result2 = mtr_asym_curve(Z2)
        self.assertTrue(result1.sum() > result2.sum())

    def test_mtr_asym(self):
        # Test MTR asymmetry image with a Lorentzian curve centered at positive ppm
        Z = generate_Z_3D(img_size=(2, 2), dyn=5, ppm=3, a=1, b=1, c=3)
        mask = np.array([[1, 1], [0, 1]])
        curve, mtr_asym_img = mtr_asym(Z, mask, (0, 1), 3)
        self.assertTrue(mtr_asym_img[mask != 0].sum() > 0)
        self.assertEqual(str(mtr_asym_img[1, 0]), "nan")

        # Test MTR asymmetry image with a Lorentzian curve centered at negative ppm
        Z = generate_Z_3D(img_size=(2, 2), dyn=25, ppm=3, a=1, b=-1, c=3)
        mask = np.array([[1, 1], [1, 1]])
        curve, mtr_asym_img = mtr_asym(Z, mask, (0, 1), 3)
        self.assertTrue(mtr_asym_img.sum() < 0)

        # Test MTR asymmetry image with a Lorentzian curve with a lower height
        Z1 = generate_Z_3D(img_size=(2, 2), dyn=5, ppm=3, a=1, b=1, c=3)
        curve1, mtr_asym_img1 = mtr_asym(Z1, mask, (0, 1), 3)
        Z2 = generate_Z_3D(img_size=(2, 2), dyn=5, ppm=3, a=0.5, b=1, c=3)
        curve2, mtr_asym_img2 = mtr_asym(Z2, mask, (0, 1), 3)
        self.assertTrue(mtr_asym_img1.sum() > mtr_asym_img2.sum())

        # Test MTR asymmetry image with a Lorentzian curve with a higher width
        Z1 = generate_Z_3D(img_size=(2, 2), dyn=5, ppm=3, a=1, b=1, c=3)
        curve1, mtr_asym_img1 = mtr_asym(Z1, mask, (0, 1), 3)
        Z2 = generate_Z_3D(img_size=(2, 2), dyn=5, ppm=3, a=1, b=1, c=6)
        curve2, mtr_asym_img2 = mtr_asym(Z2, mask, (0, 1), 3)
        self.assertTrue(mtr_asym_img1.sum() > mtr_asym_img2.sum())


if __name__ == "__main__":
    unittest.main()
