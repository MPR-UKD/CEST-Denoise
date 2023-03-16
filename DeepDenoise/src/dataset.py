import os
from pathlib import Path

import nibabel as nib
import torch
from torch.utils.data import Dataset

from Transform.src import Noiser


class CESTDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        mode: str,
        distribution: list | None = None,
        noise_std: float = 0.1,
        transform=None,
    ):

        # If distribution parameter is not provided, set default values
        distribution = distribution if distribution is not None else [0.7, 0.2, 0.1]

        # Initialize class variables
        self.mode = mode
        self.root_dir = root_dir
        self.noiser = Noiser(sigma=noise_std)
        self.transform = transform

        # Get all nii files in root directory
        files = [_.absolute() for _ in root_dir.glob("*.nii")]

        # Set the start and end index for the file list based on mode
        if mode == "train":
            start = 0
            end = int(distribution[0] * len(files))
        elif mode == "val":
            start = int(distribution[0] * len(files)) - 1
            end = start + int(distribution[1] * len(files))
        elif mode == "test":
            start = (
                int(distribution[0] * len(files))
                + int(distribution[1] * len(files))
                - 1
            )
            end = -1
        else:
            # Raise an error if mode is invalid
            raise ValueError(f"Invalid mode: {mode}")

        # Set the file list based on start and end indices
        self.file_list = files[start:end]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # If idx is a tensor, convert it to a Python integer
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the image at the given index
        img_path = self.file_list[idx]
        img_nii = nib.load(img_path)
        img = img_nii.get_fdata()

        # Add noise to the image using the Noiser class
        noisy_img = self.noiser.add_noise(img)

        # Transpose the image arrays to match PyTorch's convention
        sample = {
            "ground_truth": img.transpose(2, 0, 1, 3),
            "noisy": noisy_img.transpose(2, 0, 1, 3),
        }

        # Apply transform (if provided)
        if self.transform:
            sample = self.transform(sample)

        # Convert numpy arrays to PyTorch tensors
        sample["ground_truth"] = torch.tensor(sample["ground_truth"])
        sample["noisy"] = torch.tensor(sample["noisy"])

        return sample
