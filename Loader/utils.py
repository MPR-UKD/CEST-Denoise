import itertools
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def load_nii(file: Path) -> np.ndarray:
    return nib.load(file).get_fdata()


def load_z_spectra(file: Path) -> np.ndarray:
    img = load_nii(file)
    S0 = img[:, :, 0]
    Z = img[:, :, 1:]
    for x, y in itertools.product(range(img.shape[0]), range(img.shape[1])):
        Z[x, y] = Z[x, y] / (S0[x, y] + 0.000001)
    return Z


def get_files(root: Path, pattern: str = None) -> list[Path]:
    images = []
    for parent, dirs, files in os.walk(root):
        for file in files:
            if pattern in file:
                images.append(Path(parent) / file)
    return images
