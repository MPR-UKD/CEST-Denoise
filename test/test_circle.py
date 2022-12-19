import time

from Loader.src.loader import LoaderCEST
from PCA import pca
from NLM import nlm_CEST
from BM3D import bm3d_CEST
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).parent / "circles"
    loader = LoaderCEST(path=root)

    Z, mask, file = loader.__getitem__(0)

    start_pca = time.time()
    Z_pca = pca(Z, "nelson")
    end_pca = time.time()
    start_bm3d = end_pca
    Z_bm3d = bm3d_CEST(Z, None, True)
    end_bm3d = time.time()
    start_nlm = end_bm3d
    Z_nlm = nlm_CEST(Z, 20, 8, False)
    end_nlm = time.time()
    print(f"PCA duration: {end_pca - start_pca}")
    print(f"BM3D duration: {end_bm3d - start_bm3d}")
    print(f"NLM duration: {end_nlm - start_nlm}")
