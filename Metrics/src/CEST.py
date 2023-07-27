import numpy as np
from scipy.optimize import curve_fit
from Metrics.src.utils import mtr_asym_curve, LorentzianPool, multi_lorentzian


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


def multi_lorentzian_fit(
    Z: np.ndarray, mask: np.ndarray, ppm: float, pools: list[LorentzianPool]
) -> dict:
    """
    Perform Multi-Lorentzian fit for each pool and return the resulting images.

    Parameters:
    - Z: 3D numpy array of MTR data, with shape (num_rows, num_cols, num_dynamics)
    - mask: 2D numpy array of mask data, with shape (num_rows, num_cols)
    - ppm: float, specifying the range of ppm values for the MTR data
    - pools: list of LorentzianPool objects, specifying the pools and their bounds

    Returns:
    - Dictionary containing:
      - Multi-Lorentzian fit images for amplitude, width and position of each pool, with shape (num_rows, num_cols)
    """
    num_rows, num_cols, num_dynamics = Z.shape
    ppm_values = np.linspace(-ppm, ppm, num_dynamics)
    result_images = {
        pool.name: {
            "amplitude": np.zeros((num_rows, num_cols)),
            "width": np.zeros((num_rows, num_cols)),
            "position": np.zeros((num_rows, num_cols)),
        }
        for pool in pools
    }

    for row in range(num_rows):
        for col in range(num_cols):
            if mask[row, col] != 0:
                y_data = Z[row, col, :]
                p0 = []
                bounds_lower = []
                bounds_upper = []

                for pool in pools:
                    A0 = pool.amplitude_bounds[1]
                    x0 = pool.position_bounds[1]
                    gamma0 = pool.width_bounds[1]
                    p0.extend([A0, x0, gamma0])

                    bounds_lower.extend(
                        [
                            pool.amplitude_bounds[0],
                            pool.position_bounds[0],
                            pool.width_bounds[0],
                        ]
                    )
                    bounds_upper.extend(
                        [
                            pool.amplitude_bounds[2],
                            pool.position_bounds[2],
                            pool.width_bounds[2],
                        ]
                    )

                bounds = (bounds_lower, bounds_upper)

                try:
                    popt, _ = curve_fit(
                        multi_lorentzian, ppm_values, y_data, p0=p0, bounds=bounds
                    )
                    for i, pool in enumerate(pools):
                        result_images[pool.name]["amplitude"][row, col] = popt[i * 3]
                        result_images[pool.name]["position"][row, col] = popt[i * 3 + 1]
                        result_images[pool.name]["width"][row, col] = popt[i * 3 + 2]
                except RuntimeError:
                    for pool in pools:
                        result_images[pool.name]["amplitude"][row, col] = np.nan
                        result_images[pool.name]["position"][row, col] = np.nan
                        result_images[pool.name]["width"][row, col] = np.nan

    return result_images
