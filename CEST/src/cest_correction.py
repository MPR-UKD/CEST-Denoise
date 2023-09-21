import numpy as np

from CEST.src import matlab_style_functions


def cest_correction(cest_array: np.ndarray,
                    x_calcentires: np.ndarray,
                    x: np.ndarray,
                    x_itp: np.ndarray,
                    mask: np.ndarray,
                    offset_map: np.ndarray,
                    interpolation: str,
                    cest_range: float) -> tuple:
    """
    Correct the CEST data.

    Args:
        cest_array (np.ndarray): Array containing the CEST data.
        x_calcentires (np.ndarray): Array containing the corrected x-axis values for the CEST data.
        x (np.ndarray): Array containing the original x-axis values for the CEST data.
        x_itp (np.ndarray): Array containing the x-axis values to use for interpolation.
        mask (np.ndarray): Array containing a mask for the CEST data.
        offset_map (np.ndarray): Array containing the offset map for the WASSR data.
        interpolation (str): String indicating the interpolation method to use.
        cest_range (float): Float indicating the range of the CEST data.

    Returns:
        tuple: Corrected CEST curve and calculated x-axis values.
    """
    rows, columns, dyn = cest_array.shape
    CestCurveS = np.zeros((rows, columns, len(x_calcentires)))

    arguments = np.argwhere(mask != 0)

    for i, j in arguments:
        values = cest_array[i, j, :]
        offset = offset_map[i, j]
        CestCurveS[i, j, :] = calc_pixel(cest_range, values, offset, x_itp, x, interpolation)

    return CestCurveS, x_calcentires


def calc_pixel(cest_range: float, y_values: np.ndarray, offset: float,
               x_itp: np.ndarray, x: np.ndarray, interpolation: str) -> np.ndarray:
    """
    Correct the CEST data for a single pixel.

    Args:
        cest_range (float): The range of the CEST data.
        y_values (np.ndarray): The CEST data for the current pixel.
        offset (float): The offset map value for the current pixel.
        x_itp (np.ndarray): Array containing the x-axis values to use for interpolation.
        x (np.ndarray): Array containing the original x-axis values for the CEST data.
        interpolation (str): String indicating the interpolation method to use.

    Returns:
        np.ndarray: Corrected CEST data for a single pixel.
    """
    y_itp = matlab_style_functions.interpolate(x, y_values, x_itp, interpolation)

    vind_sC_1 = abs(x_itp - (-cest_range + offset))
    vind_sC_2 = abs(x_itp - (cest_range + offset))
    ind_sC_1 = np.argmin(vind_sC_1)
    ind_sC_2 = np.argmin(vind_sC_2)

    y_calcentries = y_itp[ind_sC_1:ind_sC_2 + 1]
    return y_calcentries
