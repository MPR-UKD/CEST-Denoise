# Import necessary libraries
import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from nibabel import load, save, Nifti1Image

from BM3D.src.denoise import bm3d, bm3d_CEST
from NLM.src.denoise import nlm, nlm_CEST
from PCA.src.denoise import pca

root = Path(__file__).parent

# Global configurations for default YAML paths
DEFAULT_CONFIGS = {
    "BM3D": root / "default_yamls/bm3d.yaml",
    "NLM": root / "default_yamls/nlm.yaml",
    "PCA": root / "default_yamls/PCA.yaml",
    "ONNX": root / "default_yamls/ONNX.yaml",
}

# Initialize the logger for logging information and errors
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def denoise_onnx(input_image: np.ndarray, model_path: str) -> np.ndarray:
    """
    Denoise an image using an ONNX model.

    Args:
    - input_image (np.ndarray): The input image to be denoised.
    - model_path (str): Path to the ONNX model.

    Returns:
    - np.ndarray: The denoised image.
    """
    ort_session = ort.InferenceSession(model_path)
    input_data = np.expand_dims(input_image.squeeze(), axis=0)
    input_name = ort_session.get_inputs()[0].name
    denoised_image = ort_session.run(None, {input_name: input_data.astype("float32")})[
        0
    ]
    return denoised_image.squeeze()


def denoise_nifti(
    input_path: str,
    output_path: str,
    algorithm: str,
    mask_path: str = None,
    config: str = None,
    onnx_model: str = None,
):
    """
    Denoise a NIFTI image using the specified algorithm.

    Args:
    - input_path (str): Path to the input NIFTI image.
    - output_path (str): Path to save the denoised NIFTI image.
    - algorithm (str): Denoising algorithm to use.
    - mask_path (str, optional): Path to a mask NIFTI image.
    - config (str, optional): Path to a yaml file with configuration parameters.
    - onnx_model (str, optional): Path to the ONNX model for denoising.

    Returns:
    - None
    """
    # Load the input image and mask (if provided)
    input_image = load(input_path).get_fdata()
    mask = load(mask_path).get_fdata() if mask_path else None

    # Load the configuration parameters from the YAML file (if provided)
    if config:
        with open(config, "r") as f:
            config = yaml.safe_load(f)
    else:
        with open(DEFAULT_CONFIGS[algorithm], "r") as f:
            config = yaml.safe_load(f)

    # Denoise the image based on the specified algorithm
    if algorithm == "ONNX":
        if not onnx_model:
            raise ValueError("ONNX model path must be provided for ONNX denoising.")
        if not Path(onnx_model).exists():
            raise ValueError("ONNX path doesn't exist!")
        denoised_image = denoise_onnx(input_image, onnx_model)
    elif algorithm == "BM3D":
        denoised_image = (
            bm3d(input_image, config=config, mask=mask)
            if config["mode"] == "image"
            else bm3d_CEST(input_image, config=config, mask=mask, multi_processing=True)
        )
    elif algorithm == "NLM":
        if config["mode"] == "image":
            denoised_image = nlm(
                input_image,
                big_window_size=config.big_window_size,
                small_window_size=config.small_window_size,
            )
        elif config["mode"] == "CEST":
            # Handle 4D Nifti images
            if len(input_image.shape) == 4:
                denoised_image = nlm_CEST(
                    input_image[:, :, 0, :],
                    big_window_size=config["big_window_size"],
                    small_window_size=config["small_window_size"],
                    multi_processing=True,
                    pools=42,
                ).reshape(input_image.shape)
            else:
                denoised_image = nlm_CEST(
                    input_image,
                    big_window_size=config["big_window_size"],
                    small_window_size=config["small_window_size"],
                    multi_processing=True,
                )
    elif algorithm == "PCA":
        if config["mode"] == "CEST":
            # Handle 4D Nifti images
            if len(input_image.shape) == 4:
                denoised_image = pca(
                    input_image[:, :, 0, :], criteria=config["criteria"], mask=mask
                ).reshape(input_image.shape)
            else:
                denoised_image = pca(
                    input_image, criteria=config["criteria"], mask=mask
                )
        else:
            raise ValueError("Invalid mode for PCA algorithm.")

    # Save the denoised image as a NIFTI file
    denoised_image = Nifti1Image(denoised_image, affine=np.eye(4))
    try:
        save(denoised_image, output_path)
        logging.info(f"Saved denoised image to {output_path} using {algorithm}")
    except Exception as e:
        logging.error(f"Error while saving the denoised image: {e}")

    # Check if the file exists after saving
    if not Path(output_path).exists():
        logging.warning(f"File {output_path} not found after saving. Retrying...")
        save(denoised_image, output_path)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Denoise a NIFTI image using BM3D, NLM, PCA, or ONNX"
    )
    parser.add_argument("input_path", type=str, help="Path to the input NIFTI image")
    parser.add_argument(
        "output_path", type=str, help="Path to save the denoised NIFTI image"
    )
    parser.add_argument(
        "algorithm",
        type=str,
        choices=["BM3D", "NLM", "PCA", "ONNX"],
        help="Denoising algorithm to use",
        default="NLM",
    )
    parser.add_argument("--mask_path", type=str, help="Path to a mask NIFTI image")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a yaml file with configuration parameters for the denoising algorithm.",
    )
    parser.add_argument(
        "--onnx_model",
        type=str,
        help="Path to the ONNX model for denoising (required if algorithm is ONNX)",
    )

    args = parser.parse_args()

    # Call the denoise function with parsed arguments
    denoise_nifti(
        args.input_path,
        args.output_path,
        args.algorithm,
        mask_path=args.mask_path,
        config=args.config,
        onnx_model=args.onnx_model,
    )
