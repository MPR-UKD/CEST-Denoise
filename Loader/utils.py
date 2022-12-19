import itertools
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def load_nii(file: Path) -> np.ndarray:
    """
    Loads the data from a NIfTI file.

    Parameters:
        file (Path): The path to the NIfTI file.

    Returns:
        np.ndarray: The data from the NIfTI file.
    """
    # Open the file and get the data
    with nib.openfile(file) as f:
        data = f.get_fdata()
    # Return the data
    return data


def load_z_spectra(file: Path) -> np.ndarray:
    """
    Loads the Z spectra from a NIfTI file.

    Parameters:
        file (Path): The path to the NIfTI file.

    Returns:
        np.ndarray: The Z spectra data.
    """
    # Load the data from the NIfTI file
    img = load_nii(file)
    # Get the S0 data and the Z spectra data
    S0 = img[:, :, 0]
    Z = img[:, :, 1:]
    # Loop through all the pixels in the image
    for x, y in itertools.product(range(img.shape[0]), range(img.shape[1])):
        # Normalize the Z spectra data using the S0 data
        Z[x, y] = Z[x, y] / (S0[x, y] + 0.000001)
    # Return the Z spectra data
    return Z


def get_files(root: Path, pattern: str = None) -> list[Path]:
    """
    Gets a list of files in the specified root directory that match the given pattern.

    Parameters:
        root (Path): The root directory to search for files.
        pattern (str, optional): The pattern to match the file names against. If not provided, all files are returned.

    Returns:
        list[Path]: A list of Path objects for the matching files.
    """
    # Initialize an empty list to store the file paths
    images = []
    # Walk through the root directory and its subdirectories
    for parent, dirs, files in os.walk(root):
        # Loop through all the files in the current directory
        for file in files:
            # If a pattern is provided, skip the file if it doesn't match the pattern
            if pattern not in file:
                continue
            # Add the file path to the list
            images.append(Path(parent) / file)
    # Return the list of file paths
    return images
