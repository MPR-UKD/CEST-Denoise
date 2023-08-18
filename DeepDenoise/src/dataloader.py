import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
from DeepDenoise.src.dataset import CESTDataset
from typing import Callable, Union, List, Optional


def get_dataset(
    dir: Union[str, Path],
    mode: str,
    distribution: Optional[List[float]],
    noise_std: float,
    transform: Optional[Callable],
    dyn: int,
    variable_sigma: bool = False,
) -> CESTDataset:
    """
    Create and return a CESTDataset.

    Args:
        dir (Union[str, Path]): Directory containing the .nii files.
        mode (str): Mode of operation ("train", "val", or "test").
        distribution (Optional[List[float]]): Data distribution across training, validation, and testing.
        noise_std (float): Standard deviation of the noise to add.
        transform (Optional[Callable]): Optional transformation to apply to the data.
        dyn (int): Number of offset frequencies in the Z-spectrum.

    Returns:
        CESTDataset: The created dataset.
    """
    return CESTDataset(
        root_dir=Path(dir),
        mode=mode,
        distribution=distribution,
        noise_std=noise_std,
        transform=transform,
        dyn=dyn,
        variable_sigma=variable_sigma,
    )


class CESTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dir: Union[str, Path],
        distribution: Optional[List[float]] = None,
        batch_size: int = 1,
        workers: int = 1,
        transform: Optional[Callable] = None,
        noise_std: float = 0.2,
        dyn: int = 42,
    ):
        """
        Initialize the CESTDataModule class.

        Args:
            dir (Union[str, Path]): Directory containing the .nii files.
            distribution (Optional[List[float]]): Data distribution across training, validation, and testing.
            batch_size (int): Batch size for data loading.
            workers (int): Number of workers for data loading.
            transform (Optional[Callable]): Optional transformation to apply to the data.
            noise_std (float): Standard deviation of the noise to add.
            dyn (int): Number of offset frequencies in the Z-spectrum.
        """
        super().__init__()
        self.dir = dir
        self.train_dataset = get_dataset(
            dir, "train", distribution, noise_std + 0.05, transform, dyn, True
        )
        self.val_dataset = get_dataset(
            dir, "val", distribution, noise_std, transform, dyn
        )
        self.test_dataset = get_dataset(
            dir, "test", distribution, noise_std, transform, dyn
        )
        self.batch_size = batch_size
        self.workers = workers

    def train_dataloader(self) -> DataLoader:
        """Return a DataLoader for the training dataset."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return a DataLoader for the validation dataset."""
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.workers
        )

    def test_dataloader(self) -> DataLoader:
        """Return a DataLoader for the testing dataset."""
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.workers
        )
