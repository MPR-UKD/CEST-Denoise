import time

from Loader.loader import LoaderCEST
from PCA import pca
from NLM import nlm
from BM3D import bm3d
from pathlib import Path


if __name__ == '__main__':
    root = Path(__file__).parent / 'circles'
    loader = LoaderCEST(path=root)

    Z, mask, file = loader.__getitem__(0)

    _ = bm3d((Z[:64, :64, 0] * 255).astype('int16'))
    b = 2