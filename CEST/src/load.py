from pathlib import Path

import nibabel as nib
import numpy as np


def load_norm_cest_data(cest_file, mask_file: Path | None = None):
    img_nii = nib.load(cest_file)
    img = img_nii.get_fdata().squeeze()
    cest = np.zeros((img.shape[0], img.shape[1], img.shape[2] - 1))
    if mask_file is not None:
        mask_nii = nib.load(mask_file)
        if len(mask_nii.get_fdata().squeeze().shape) == 3:
            mask = np.max(mask_nii.get_fdata().squeeze(), axis=-1)
        else:
            mask = mask_nii.get_fdata().squeeze()
    else:
        mask = np.ones(img.shape[:2])

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y, 0] == 0:
                img[x, y, 0] = 0.00001

            cest[x, y, :] = img[x, y, 1:] / img[x, y, 0]
    cest[cest > 1.2] = 1
    return cest, mask
