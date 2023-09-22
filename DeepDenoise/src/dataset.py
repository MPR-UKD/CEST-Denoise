import random
from pathlib import Path
from typing import List, Dict, Union, Optional, Callable

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from Transform.src import Noiser


class CESTDataset(Dataset):
    """A Dataset class for handling CEST MRI data."""

    def __init__(self,
                 root_dir: Path,
                 mode: str,
                 distribution: Optional[List[float]] = None,
                 noise_std: float = 0.1,
                 transform: Optional[Callable] = None,
                 dyn: Optional[int] = None,
                 variable_sigma: bool = False):
        """
        Initialize the CESTDataset class.

        Args:
            root_dir (Path): The root directory containing the .nii files.
            mode (str): The mode in which to operate, either "train", "val", or "test".
            distribution (List[float], optional): Data distribution across training, validation, and testing.
                Defaults to [0.7, 0.2, 0.1].
            noise_std (float, optional): Standard deviation of the noise to add. Defaults to 0.1.
            transform (Callable, optional): Optional transformation to apply to the data.
            dyn (int, optional): Number of offset frequencies in the Z-spectrum.
            variable_sigma (bool, optional): Whether to use a variable sigma for noise addition. Defaults to False.

        Raises:
            ValueError: If an invalid mode is provided.
        """
        # Set default distribution if not provided
        distribution = distribution if distribution else [0.7, 0.2, 0.1]

        self.mode = mode
        self.root_dir = root_dir
        self.noiser = Noiser(sigma=noise_std)
        self.transform = transform
        self.dyn = dyn
        self.sigma = noise_std
        self.variable_sigma = variable_sigma

        # Get all .nii files in the root directory
        files = [file.absolute() for file in root_dir.glob("*.nii")]

        # Determine start and end indices based on mode and distribution
        total_files = len(files)
        if mode == "train":
            start, end = 0, int(distribution[0] * total_files)
        elif mode == "val":
            start, end = int(distribution[0] * total_files), int(
                distribution[0] * total_files + distribution[1] * total_files)
        elif mode == "test":
            start, end = int(distribution[0] * total_files + distribution[1] * total_files), total_files
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.file_list = files[start:end]

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.file_list)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Retrieve an item from the dataset by index.

        Args:
            idx (Union[int, torch.Tensor]): Index of the desired item.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing ground truth and noisy images.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.file_list[idx]
        img = load_z(img_path, self.dyn)

        noisy_img = self.noiser.add_noise(img.copy(),
                                          sigma=random.uniform(0, self.sigma) if self.variable_sigma else None)

        # Transpose to match PyTorch's convention (C, H, W)
        sample = {
            "ground_truth": torch.tensor(img.transpose(2, 3, 0, 1).squeeze(), dtype=torch.float32),
            "noisy": torch.tensor(noisy_img.transpose(2, 3, 0, 1).squeeze(), dtype=torch.float32),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def load_z(img_path: Path, dyn: Optional[int] = None) -> np.ndarray:
    """
    Load a CEST file and normalize it.

    Args:
        img_path (Path): Path to the .nii file.
        dyn (int, optional): Number of offset frequencies in the Z-spectrum.

    Returns:
        np.ndarray: Normalized image data.
    """
    img_nii = nib.load(img_path)
    img = img_nii.get_fdata()

    # Normalize the image
    if img.max() > 2:
        img /= 4016

    # Crop the Z-spectrum if dyn is provided
    if dyn:
        first_offset = random.randint(0, img.shape[-1] - dyn)
        img = img[:, :, :, first_offset: first_offset + dyn]

    return img
