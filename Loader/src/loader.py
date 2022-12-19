import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path

from Loader.src.utils import load_nii, load_z_spectra, get_files


class Loader(ABC):
    def __init__(self, path: Path):
        path = Path(path)
        file_mode = True if path.is_file() else False

        if file_mode:
            self.files = [path]
        else:
            self.files = get_files(path, "image.nii")

    def __len__(self):
        return len(self.files)

    @abstractmethod
    def __getitem__(self, idx):
        pass


class LoaderCEST(Loader):
    def __init__(self, path: Path):
        super().__init__(path)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, Path]:
        file = self.files[idx]
        mask_file = file.parent / "mask.nii.gz"
        return load_z_spectra(file), load_nii(mask_file), file
