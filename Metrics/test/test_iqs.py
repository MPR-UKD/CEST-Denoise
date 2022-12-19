import unittest
import numpy as np

from Metrics.src.image_quality_estimation import IQS


class TestIQS(unittest.TestCase):
    def setUp(self):
        # Create a reference image for testing
        self.ref_image = np.ones((10, 10))
        # Create an instance of the IQS class
        self.iqs = IQS(pixel_max=255, ref_image=self.ref_image)


    def test_mse(self):
        # Create a test image with the same shape as the reference image
        test_image = np.zeros((10, 10))
        # The MSE between the test image and the reference image should be 1
        self.assertEqual(self.iqs.mse(test_image), 1)

        # Create a test image with a different shape than the reference image
        test_image = np.zeros((5, 5))
        # The MSE between the test image and the reference image should be 1
        self.assertRaises(ValueError, self.iqs.mse, test_image)

    def test_psnr(self):
        # Create a test image with the same shape as the reference image
        test_image = np.zeros((10, 10))
        # The PSNR between the test image and the reference image should be around 41.4
        self.assertAlmostEqual(self.iqs.psnr(test_image), 48.1, delta=0.1)

        # Create a test image with a different shape than the reference image
        test_image = np.zeros((5, 5))
        # The MSE between the test image and the reference image should be 1
        self.assertRaises(ValueError, self.iqs.psnr, test_image)

    def test_root_mean_square_error(self):
        # Create a test image with the same shape as the reference image
        test_image = np.zeros((10, 10))
        # The root mean squared error between the test image and the reference image should be 1
        self.assertEqual(self.iqs.root_mean_square_error(test_image), 1)

        # Create a test image with a different shape than the reference image
        test_image = np.zeros((5, 5))
        # The root mean squared error between the test image and the reference image should be 1
        self.assertRaises(ValueError, self.iqs.root_mean_square_error, test_image)


if __name__ == '__main__':
    unittest.main()
