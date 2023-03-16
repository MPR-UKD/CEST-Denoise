import nibabel as nib
from skimage.util import random_noise

from PIL import Image
import numpy as np
from pathlib import Path
from Transform.src.noise import Noiser


def add_noise_and_save(output_path, sigma):
    # Load the default "Lena" image using Pillow
    image = Image.open("default.jpg").convert("L")

    # Convert the image to a NumPy array
    data = np.array(image).astype("float64")
    data /= 255.0

    # Add noise to the image using skimage
    noiser = Noiser(sigma=sigma)
    noisy_data = noiser.add_noise(data) * 255.0

    # Create a new NIFTI image with the noisy data
    noisy_image = nib.Nifti1Image(noisy_data, np.eye(4), nib.Nifti1Header())

    # Save the noisy image to the output path
    nib.save(noisy_image, output_path)

    # Save the noisy image as png
    noisy_image = Image.fromarray(noisy_data).convert("RGB")
    noisy_image.save(output_path.as_posix().replace("nii.gz", "png"))


if __name__ == "__main__":
    base = Path(__file__).parent
    add_noise_and_save(base / "image_sigma_0.nii.gz", sigma=0)
    add_noise_and_save(base / "image_sigma_0_025.nii.gz", sigma=0.025)
    add_noise_and_save(base / "image_sigma_0_05.nii.gz", sigma=0.05)
    add_noise_and_save(base / "image_sigma_0_1.nii.gz", sigma=0.1)
