import argparse

from nibabel import load, save, Nifti1Image

from BM3D.src.denoise import bm3d, bm3d_CEST
from NLM.src.denoise import nlm, nlm_CEST
from PCA.src.denoise import pca
import yaml
import numpy as np


def denoise_nifti(input_path, output_path, algorithm, mask_path=None, config=None):
    # Load the input image and mask (if provided)
    input_image = load(input_path).get_fdata()
    if mask_path:
        mask = load(mask_path).get_fdata()
    else:
        mask = None

    # Load the configuration parameters from the YAML file (if provided)
    if config:
        with open(config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "BM3D": "default_yamls/bm3d.yaml",
            "NLM": "default_yamls/nlm.yaml",
            "PCA": "default_yamls/PCA.yaml",
        }
        with open(config[algorithm], "r") as f:
            config = yaml.safe_load(f)

    # Denoise the image using the specified algorithm
    if algorithm == "BM3D":
        if config.mode == "image":
            denoised_image = bm3d(input_image, config=config, mask=mask)
        elif config.mode == "cest":
            denoised_image = bm3d_CEST(input_image, config=config, mask=mask)
        else:
            raise ValueError
    elif algorithm == "NLM":
        if config.mode == "image":
            denoised_image = nlm(
                input_image,
                big_window_size=config.big_window_size,
                small_window_size=config.small_window_size,
            )
        elif config.mode == "cest":
            denoised_image = nlm_CEST(
                input_image,
                big_window_size=config.big_window_size,
                small_window_size=config.small_window_size,
            )
        else:
            raise ValueError
    elif algorithm == "PCA":
        if config.mode == "cest":
            denoised_image = pca(input_image, criteria=config.criteria, mask=mask)
        else:
            raise ValueError

    # Save the denoised image as a NIFTI file
    denoised_image = Nifti1Image(denoised_image, affine=np.eye(4))
    save(denoised_image, output_path)


# Parse the command-line arguments
parser = argparse.ArgumentParser(
    description="Denoise a NIFTI image using BM3D, NLM, or PCA"
)
parser.add_argument("input_path", type=str, help="Path to the input NIFTI image")
parser.add_argument(
    "output_path", type=str, help="Path to save the denoised NIFTI image"
)
parser.add_argument(
    "algorithm",
    type=str,
    choices=["BM3D", "NLM", "PCA"],
    help="Denoising algorithm to use (default: NLM)",
    default="NLM",
)
parser.add_argument("--mask_path", type=str, help="Path to a mask NIFTI image")
parser.add_argument(
    "--config",
    type=str,
    help="path to a yaml file with configuration parameters for the denoising algorithm. If no path "
    "is specified, the default config is used.",
)
args = parser.parse_args()

# Denoise the NIFTI image
denoise_nifti(
    args.input_path,
    args.output_path,
    args.algorithm,
    mask_path=args.mask_path,
    config=args.config,
)
