import tempfile
import unittest

import nibabel as nib

from Loader.src.loader import *


class TestLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the test data
        self.temp_dir = tempfile.TemporaryDirectory()
        # Create a dummy image file for testing
        self.image_file = Path(self.temp_dir.name) / "image.nii"
        img = np.zeros((10, 10, 10))
        nib.save(nib.Nifti1Image(img, np.eye(4)), self.image_file)
        # Create a dummy mask file for testing
        self.mask_file = Path(self.temp_dir.name) / "mask.nii.gz"
        img = np.ones((10, 10, 10))
        nib.save(nib.Nifti1Image(img, np.eye(4)), self.mask_file)
        # Create a dummy LoaderCEST instance for testing
        self.loader = LoaderNiftiCEST(self.image_file)

    def test_len(self):
        # Test the __len__ method of the LoaderCEST class
        self.assertEqual(len(self.loader), 1)

    def test_getitem(self):
        # Test the __getitem__ method of the LoaderCEST class
        spectra, mask, file = self.loader[0]
        # Assert that the returned values are NumPy arrays
        self.assertIsInstance(spectra, np.ndarray)
        self.assertIsInstance(mask, np.ndarray)
        # Assert that the returned path is a Path object
        self.assertIsInstance(file, Path)
        # Assert that the returned path is the correct path
        self.assertEqual(file, self.image_file)


if __name__ == "__main__":
    unittest.main()
