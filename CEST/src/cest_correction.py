import numpy as np

# Import the matlab_style_functions module from the CEST package
from CEST.src import matlab_style_functions


def cest_correction(
    cest_array: np.ndarray,  # numpy array containing the CEST data
    x_calcentires: np.ndarray,  # numpy array containing the corrected x-axis values for the CEST data
    x: np.ndarray,  # numpy array containing the original x-axis values for the CEST data
    x_itp: np.ndarray,  # numpy array containing the x-axis values to use for interpolation
    mask: np.ndarray,  # numpy array containing a mask for the CEST data
    offset_map: np.ndarray,  # numpy array containing the offset map for the WASSR data
    interpolation: str,  # string indicating the interpolation method to use
    cest_range,  # float indicating the range of the CEST data
):
    # Get the shape of the CEST data array
    (rows, colums, dyn) = cest_array.shape
    # Initialize an array to store the corrected CEST data
    CestCurveS = np.zeros((rows, colums, len(x_calcentires)))

    # Find the indices of the non-zero elements in the mask
    arguments = np.argwhere(mask != 0)
    # Iterate over the indices
    for i, j in arguments:
        # Get the CEST data and offset map value for the current pixel
        values = cest_array[i, j, :]
        offset = offset_map[i, j]
        # Correct the CEST data for the current pixel
        CestCurveS[i, j, :] = calc_pixel(
            cest_range, values, offset, x_itp, x, interpolation
        )

    return CestCurveS, x_calcentires


def calc_pixel(range, y_values, offset, x_itp, x, interpolation):
    """Correct the CEST data for a single pixel."""
    # Interpolate the CEST data for the current pixel
    y_itp = matlab_style_functions.interpolate(x, y_values, x_itp, interpolation)

    # Find the indices of the start and end points of the corrected CEST data range
    vind_sC_1 = abs(x_itp - (-range + offset))
    vind_sC_2 = abs(x_itp - (range + offset))
    ind_sC_1 = np.argmin(vind_sC_1)
    ind_sC_2 = np.argmin(vind_sC_2)

    y_calcentries = y_itp[ind_sC_1 : ind_sC_2 + 1]
    return y_calcentries
