import unittest
from CEST.src.WASSR import WASSR
import numpy as np
from test_support_function.src.CEST import generate_Z_3D


class TestWASSR(unittest.TestCase):
    def test_OF_0(self):
        # Test WASSR with a Lorentzian curve centered at 0 ppm
        Z = generate_Z_3D(img_size=(2, 2), dyn=21, ppm=3, a=1, b=0, c=3)
        wassr = WASSR(1, 3)
        offset_map, mask = wassr.calculate(wassr=Z, mask=np.ones((2, 2)), hStep=0.01)
        self.assertAlmostEqual(offset_map[0, 0], 0, delta=0.01)

    def test_OF_0_5(self):
        # Test WASSR with a Lorentzian curve centered at 0.5 ppm
        Z = generate_Z_3D(img_size=(2, 2), dyn=21, ppm=3, a=1, b=0.5, c=3)
        wassr = WASSR(1, 3)
        offset_map, mask = wassr.calculate(wassr=Z, mask=np.ones((2, 2)), hStep=0.01)
        self.assertAlmostEqual(offset_map[0, 0], 0.5, delta=0.01)

    def test_OF_3(self):
        # Test WASSR with a Lorentzian curve centered at 3 ppm
        Z = generate_Z_3D(img_size=(2, 2), dyn=21, ppm=3, a=1, b=3, c=3)
        wassr = WASSR(1, 3)
        offset_map, mask = wassr.calculate(wassr=Z, mask=np.ones((2, 2)), hStep=0.01)
        self.assertEqual(mask[0, 0], 0.0)

    def test_OF_negative(self):
        # Test WASSR with a Lorentzian curve centered at -0.5 ppm
        Z = generate_Z_3D(img_size=(2, 2), dyn=21, ppm=3, a=1, b=-0.5, c=3)
        wassr = WASSR(1, 3)
        offset_map, mask = wassr.calculate(wassr=Z, mask=np.ones((2, 2)), hStep=0.01)
        self.assertAlmostEqual(offset_map[0, 0], -0.5, delta=0.01)


if __name__ == "__main__":
    unittest.main()
