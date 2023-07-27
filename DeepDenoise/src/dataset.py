from pathlib import Path
from pathlib import Path
from typing import List, Dict, Union

import nibabel as nib
import torch
from torch.utils.data import Dataset

from Transform.src import Noiser


class CESTDataset(Dataset):
    """A Dataset class for handling CEST MRI data."""

    def __init__(
        self,
        root_dir: Path,
        mode: str,
        distribution: list | None = None,
        noise_std: float = 0.1,
        transform=None,
    ):
        """Init Class.

        Args:
            root_dir (Path): The root directory that contains the .nii files.
            mode (str): The mode in which to operate, either "train", "val", or "test".
            distribution (List[float] | None, optional): The data distribution across training, validation, and testing. Defaults to [0.7, 0.2, 0.1].
            noise_std (float, optional): The standard deviation of the noise to add. Defaults to 0.1.
            transform (callable, optional): An optional transformation to apply to the data.

        Raises:
            ValueError: If an invalid mode is given.

        Attributes:
            mode (str): The mode in which the dataset operates.
            root_dir (Path): The root directory containing the .nii files.
            noiser (Noiser): The Noiser object used to add noise to the images.
            transform (callable, optional): An optional transformation to apply to the data.
            file_list (List[Path]): A list of paths to the .nii files.
        """

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

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.file_list)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get an item from the dataset by index.

        Args:
            idx (int | torch.Tensor): The index of the item to get.

        Returns:
            dict[str, torch.Tensor]: The item, consisting of the ground truth and noisy images.
        """
        # If idx is a tensor, convert it to a Python integer
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the image at the given index
        img_path = self.file_list[idx]
        img = load_z(img_path)

        # Add noise to the image using the Noiser class
        noisy_img = self.noiser.add_noise(img)

        # Transpose the image arrays to match PyTorch's convention
        sample = {
            "ground_truth": img.transpose(2, 3, 0, 1).squeeze(),
            "noisy": noisy_img.transpose(2, 3, 0, 1).squeeze(),
        }

        # Apply transform (if provided)
        if self.transform:
            sample = self.transform(sample)

        # Convert numpy arrays to PyTorch tensors
        sample["ground_truth"] = torch.tensor(sample["ground_truth"]).float()
        sample["noisy"] = torch.tensor(sample["noisy"]).float()

        return sample


def load_z(img_path: Path) -> np.ndarray:
    """Load a CEST file and normalize.

    Args:
        img_path (Path): The path to the .nii file.

    Returns:
        np.ndarray: The normalized image data from the .nii file.
    """
    img_nii = nib.load(img_path)
    img = img_nii.get_fdata() / 4016
    return img
