import time

from Loader.loader import LoaderCEST
from PCA import pca
from NLM import nlm

from pathlib import Path


if __name__ == '__main__':
    root = Path(__file__).parent / 'circles'
    loader = LoaderCEST(path=root)

    Z, mask, file = loader.__getitem__(0)
    #pca(Z, 'malinowski')
    k = time.time()
    nlm(Z[:, :, 0], 16, 8)
    print(time.time() - k)