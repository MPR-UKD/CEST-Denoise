from pathlib import Path
from typing import Tuple, Union

import nibabel as nib
import numpy as np


def load_norm_cest_data(cest_file: Union[Path, str],
                        mask_file: Union[Path, str, None] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and normalize CEST data from the given file paths.

    Args:
        cest_file (Union[Path, str]): Path to the CEST Nifti file.
        mask_file (Union[Path, str, None]): Optional; Path to the Mask Nifti file.
            If not provided, a default mask of ones is used.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing normalized CEST data and the mask data.
    """
    img_nii = nib.load(cest_file)
    img = img_nii.get_fdata().squeeze()

    # Initialize CEST array
    cest = np.zeros((img.shape[0], img.shape[1], img.shape[2] - 1))

    # Load mask if provided, else create a default mask of ones
    if mask_file:
        mask_nii = nib.load(mask_file)
        mask_data = mask_nii.get_fdata().squeeze()
        mask = np.max(mask_data, axis=-1) if len(mask_data.shape) == 3 else mask_data
    else:
        mask = np.ones(img.shape[:2])

    # Normalize CEST data
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # Prevent division by zero
            img[x, y, 0] = max(img[x, y, 0], 0.00001)
            cest[x, y, :] = img[x, y, 1:] / img[x, y, 0]

    # Clamp CEST values
    cest = np.clip(cest, None, 1.2)

    return cest, mask
