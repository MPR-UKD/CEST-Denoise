# CEST Loader

This repository contains a class for loading Chemical Exchange Saturation Transfer (CEST) data stored in Nifti format.

The **'LoaderNiftiCEST'** class is a subclass of **'Loader'**, which is an abstract base class for loading data. The **'LoaderNiftiCEST'** class is created with the path to a single CEST nifti file or a directory with multiple CEST nifti files.

The length of the loader is equal to the number of files. The index operator [] can be used to access a specific CEST data sample. The sample is returned as a tuple containing the CEST data (a numpy array), the mask data (a numpy array), and the file path (a path object).

The **'load_z_spectra'** and **'load_nii'** methods are used to load the CEST data and mask data from the Nifti files. The **'get_files'** method is used to find all CEST nifti files in a directory.


## Usage

````python
from Loader.src.loader import LoaderNiftiCEST

# Create a LoaderNiftiCEST object, passing in the path to a single CEST Nifti file or a directory containing multiple CEST Nifti files
loader = LoaderNiftiCEST("path/to/file_or_directory")

# The length of the loader is equal to the number of files
print(loader.__len__())

# You can access a specific CEST data sample by index
sample = loader[0]

# The sample is returned as a tuple containing the CEST data (a numpy array), the mask data (a numpy array), and the file path (a Path object)
cest_data, mask_data, file_path = sample
````

## Usage with DICOM-Files

It is also possible to use DICOM images - for this purpose there is the support package dcm_transformer.

````python
from Loader.src.loader import LoaderNiftiCEST
import tempfile
from pathlib import Path
from Loader.src.dcm_transformer import dcm_to_nii


temp_dir = tempfile.TemporaryDirectory()

FF_CEST = Path(r"path/to/cest")
dcm_to_nii(FF_CEST, r"tmp/cest.nii.gz")

# Create a LoaderNiftiCEST object, passing in the path to a single CEST Nifti file or a directory containing multiple CEST Nifti files
loader = LoaderNiftiCEST("tmp")

# The length of the loader is equal to the number of files
print(loader.__len__())

# You can access a specific CEST data sample by index
sample = loader[0]

# The sample is returned as a tuple containing the CEST data (a numpy array), the mask data (a numpy array), and the file path (a Path object)
cest_data, mask_data, file_path = sample
````