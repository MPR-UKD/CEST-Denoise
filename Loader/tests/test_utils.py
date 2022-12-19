import tempfile
import unittest

from Loader.src.utils import *


class TestLoadFunctions(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the test data
        self.temp_dir = tempfile.TemporaryDirectory()
        # Create a dummy NIfTI file for testing
        self.nii_file = Path(self.temp_dir.name) / "test.nii"
        img = np.zeros((10, 10, 10))
        nib.save(nib.Nifti1Image(img, np.eye(4)), self.nii_file)

    def test_load_nii(self):
        # Test loading data from a NIfTI file
        data = load_nii(self.nii_file)
        # Assert that the data is a NumPy array
        self.assertIsInstance(data, np.ndarray)

    def test_load_z_spectra(self):
        # Test loading Z spectra data from a NIfTI file
        data = load_z_spectra(self.nii_file)
        # Assert that the data is a NumPy array
        self.assertIsInstance(data, np.ndarray)


class TestGetFiles(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the test data
        self.temp_dir = tempfile.TemporaryDirectory()
        # Create a dummy root directory for testing
        self.root_dir = Path(self.temp_dir.name) / "root"
        self.root_dir.mkdir()
        # Create some dummy files in the root directory
        for i in range(5):
            Path(self.root_dir / f"test{i}.txt").touch()
        for i in range(5):
            Path(self.root_dir / f"test{i}.nii").touch()

    def test_get_files(self):
        # Test getting a list of files from the root directory
        files = get_files(self.root_dir)
        # Assert that the returned value is a list of Path objects
        self.assertIsInstance(files, list)
        self.assertIsInstance(files[0], Path)
        # Assert that the list has the expected number of files
        self.assertEqual(len(files), 10)

    def test_get_files_with_pattern(self):
        # Test getting a list of files from the root directory that match a pattern
        files = get_files(self.root_dir, pattern=".nii")
        # Assert that the returned value is a list of Path objects
        self.assertIsInstance(files, list)
        self.assertIsInstance(files[0], Path)
        # Assert that the returned list only contains NIfTI files
        self.assertTrue(all(str(f).endswith(".nii") for f in files))
        # Assert that the list has the expected number of files
        self.assertEqual(len(files), 5)


if __name__ == "__main__":
    unittest.main()
