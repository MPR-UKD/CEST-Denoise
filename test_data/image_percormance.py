# Import the necessary modules and functions
from pathlib import Path

import matplotlib.pyplot as plt

from BM3D.src.denoise import bm3d
from Loader.src.utils import load_nii
from NLM.src.denoise import nlm

if __name__ == "__main__":
    # Get the base path of the current file
    base_path = Path(__file__).parent

    # Load the original image and the noisy image
    gt_img = load_nii(base_path / "images" / "image_sigma_0.nii.gz")
    noise_img = load_nii(base_path / "images" / "image_sigma_0_1.nii.gz")

    # Denoise the noisy image using the bm3d and nlm functions
    denoise_bm3d = bm3d(noise_img.astype("int16"))
    denoise_nlm = nlm(noise_img.astype("int16"), 15, 3)

    # Create a figure with a subplot grid with 2 rows and 2 columns
    fig, axs = plt.subplots(nrows=2, ncols=2)

    # Plot the original image in the top left subplot
    axs[0, 0].imshow(gt_img, cmap="gray")
    axs[0, 0].set_title("Original image")

    # Plot the noisy image in the top right subplot
    axs[0, 1].imshow(noise_img, cmap="gray")
    axs[0, 1].set_title("Noisy image")

    # Plot the denoised image using bm3d in the bottom left subplot
    axs[1, 0].imshow(denoise_bm3d, cmap="gray")
    axs[1, 0].set_title("Denoised image (bm3d)")

    # Plot the denoised image using nlm in the bottom right subplot
    axs[1, 1].imshow(denoise_nlm, cmap="gray")
    axs[1, 1].set_title("Denoised image (nlm)")

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Show the plot
    plt.show()
