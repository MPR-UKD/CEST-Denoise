import numpy as np


def mtr_asym_curve(Z):
    dyns = Z.shape[-1]
    idx1 = [_ for _ in range(int((dyns - 1) / 2), -1, -1)]
    idx2 = [_ for _ in range(int((dyns - 1) / 2), dyns, 1)]
    return  Z[:, :, idx1] - Z[:, :, idx2]


def mtr_asym(Z: np.ndarray, mask: np.ndarray, mtr_asym_ppm, ppm):
    ppm = np.linspace(-ppm, ppm, Z.shape[2])
    idx1 = np.argmin(abs(ppm - mtr_asym_ppm[0]))
    idx2 = np.argmin(abs(ppm - mtr_asym_ppm[1]))

    mtr_asym = mtr_asym_curve(Z)
    mtr_asym_img = np.sum(mtr_asym, axis=2) / mtr_asym.shape[-1] * 100
    mtr_asym_img[mask == 0] = np.nan
    return mtr_asym, mtr_asym_img
