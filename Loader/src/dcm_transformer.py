from pathlib import Path

import nibabel
import pydicom
import numpy


def dcm_to_nii(dcm_folder: Path, nii_file: Path):
    """
    Converts DICOM files in a folder to a NIfTI file.

    Parameters:
        dcm_folder (Path): The path to the folder containing the DICOM files.
        nii_file (Path): The path to the NIfTI file to be created.

    Returns:
        None
    """
    # Read the pixel arrays from the DICOM files and create a NumPy array
    dcm = numpy.array(
        [
            pydicom.dcmread(dcm_file, force=True).pixel_array
            for dcm_file in dcm_folder.glob("*.dcm")
        ]
    ).transpose(2, 1, 0)
    # Create a NIfTI image from the DICOM data
    nii = nibabel.Nifti1Image(dcm, numpy.eye(4))
    # Save the NIfTI image to the specified file
    nibabel.save(nii, nii_file)
