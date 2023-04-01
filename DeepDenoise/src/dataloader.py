import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
from DeepDenoise.src.dataset import CESTDataset
from typing import Callable


def get_dataset(
    dir: str | Path,
    mode: str,
    distribution: list | None,
    noise_std: float,
    transform: Callable | None,
):
    return CESTDataset(
        root_dir=Path(dir),
        mode=mode,
        distribution=distribution,
        noise_std=noise_std,
        transform=transform,
    )


class CESTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dir: str | Path,
        distribution: list | None = None,
        batch_size: int = 1,
        workers: int = 1,
        transform: None | Callable = None,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.dir = dir
        self.train_dataset = get_dataset(
            dir, "train", distribution, noise_std, transform
        )
        self.val_dataset = get_dataset(dir, "val", distribution, noise_std, transform)
        self.test_dataset = get_dataset(dir, "test", distribution, noise_std, transform)
        self.batch_size = batch_size
        self.workers = workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.workers)
