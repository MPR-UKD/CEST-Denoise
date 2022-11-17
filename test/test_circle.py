from Loader.loader import LoaderCEST
from PCA.src.denoise import pca
from pathlib import Path


if __name__ == '__main__':
    root = Path(__file__).parent / 'circles'
    loader = LoaderCEST(path=root)

    Z, mask, file = loader.__getitem__(0)
    pca(Z, 'median')
