import unittest
from CEST.src.matlab_style_functions import *


class TestInterpolateFunctions(unittest.TestCase):
    def test_matlab_style_gauss2D(self):
        # Test with default sigma value
        kernel = matlab_style_gauss2D((5, 5))
        expected_kernel = np.array(
            [
                [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
                [0.01330621, 0.0596343, 0.09832033, 0.0596343, 0.01330621],
                [0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823],
                [0.01330621, 0.0596343, 0.09832033, 0.0596343, 0.01330621],
                [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
            ]
        )
        np.testing.assert_almost_equal(kernel, expected_kernel, decimal=2)

        # Test with sigma = 2.0
        kernel = matlab_style_gauss2D((5, 5), sigma=2.0)
        expected_kernel = np.array(
            [
                [0.02324684, 0.03382395, 0.03832756, 0.03382395, 0.02324684],
                [0.03382395, 0.04921356, 0.05576627, 0.04921356, 0.03382395],
                [0.03832756, 0.05576627, 0.06319146, 0.05576627, 0.03832756],
                [0.03382395, 0.04921356, 0.05576627, 0.04921356, 0.03382395],
                [0.02324684, 0.03382395, 0.03832756, 0.03382395, 0.02324684],
            ]
        )
        np.testing.assert_almost_equal(kernel, expected_kernel, decimal=3)

    def test_interpolate(self):
        x = np.array([0, 1, 2, 3])
        y = np.array([0, 1, 4, 9])
        points = np.array([0.5, 1, 1.5, 2.5])

        # Test cubic interpolation
        cubic_interp = interpolate(x, y, points, "cubic")
        expected_cubic_interp = np.array([0.3, 1, 2.2, 6.2])
        np.testing.assert_almost_equal(cubic_interp, expected_cubic_interp, decimal=1)

        # Test quadratic interpolation
        quadratic_interp = interpolate(x, y, points, "quadratic")
        expected_quadratic_interp = np.array([0.3, 1, 2.2, 6.2])
        np.testing.assert_almost_equal(
            quadratic_interp, expected_quadratic_interp, decimal=1
        )

        # Test with invalid interpolation type
        with self.assertRaises(KeyError):
            interpolate(x, y, points, "linear")


if __name__ == "__main__":
    unittest.main()
