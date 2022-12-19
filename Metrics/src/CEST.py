import numpy as np


def mtr_asym_curve(Z: np.ndarray) -> np.ndarray:
    """
    Calculate the MTR asymmetry curve.

    Parameters:
    - Z: 3D numpy array of MTR data, with shape (num_rows, num_cols, num_dynamics)

    Returns:
    - MTR asymmetry curve, with shape (num_rows, num_cols, num_dynamics)
    """
    dyns = Z.shape[-1]
    idx1 = [_ for _ in range(int((dyns - 1) / 2), -1, -1)]
    idx2 = [_ for _ in range(int((dyns - 1) / 2), dyns, 1)]
    return Z[:, :, idx1] - Z[:, :, idx2]


def mtr_asym(
    Z: np.ndarray, mask: np.ndarray, mtr_asym_ppm: tuple, ppm: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate MTR asymmetry and MTR asymmetry image.

    Parameters:
    - Z: 3D numpy array of MTR data, with shape (num_rows, num_cols, num_dynamics)
    - mask: 2D numpy array of mask data, with shape (num_rows, num_cols)
    - mtr_asym_ppm: tuple of floats, specifying the range of ppm values to use for MTR asymmetry calculation
    - ppm: float, specifying the range of ppm values for the MTR data

    Returns:
    - Tuple containing:
      - MTR asymmetry curve, with shape (num_rows, num_cols, num_dynamics)
      - MTR asymmetry image, with shape (num_rows, num_cols)
    """
    ppm = np.linspace(-ppm, 0, round((Z.shape[2] / 2) + 0.0001))
    idx1 = np.argmin(abs(ppm - mtr_asym_ppm[0]))
    idx2 = np.argmin(abs(ppm - mtr_asym_ppm[1]))
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1

    mtr_asym = mtr_asym_curve(Z)
    mtr_asym_img = (
        np.sum(mtr_asym[:, :, idx1 : idx2 + 1], axis=2) / (idx2 + 1 - idx1) * 100
    )
    mtr_asym_img[mask == 0] = np.nan
    return mtr_asym, mtr_asym_img
