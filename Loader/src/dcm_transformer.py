from pathlib import Path

import nibabel
import pydicom
import numpy


def dcm_to_nii(dcm_folder: Path, nii_file: Path):
    dcm = numpy.array([pydicom.dcmread(dcm_file).pixel_array for dcm_file in dcm_folder.glob('*.dcm')]).transpose(2, 1,0)
    nii = nibabel.Nifti1Image(dcm, numpy.eye(4))
    nibabel.save(nii, nii_file)
