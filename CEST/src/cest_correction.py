import numpy as np
from CEST.src.CEST import matlab_style_functions


def cest_correction(
    cest_array: np.ndarray,
    x_calcentires: np.ndarray,
    x: np.ndarray,
    x_itp: np.ndarray,
    mask: np.ndarray,
    offset_map: np.ndarray,
    interpolation: str,
    cest_range,
    multiprocessing=False,
):
    (rows, colums, dyn) = cest_array.shape
    CestCurveS = np.zeros((rows, colums, len(x_calcentires)))

    arguments = np.argwhere(mask != 0)
    for i, j in arguments:
        values = cest_array[i, j, :]
        offset = offset_map[i, j]
        CestCurveS[i, j, :] = calc_pixel(
            cest_range, values, offset, x_itp, x, interpolation
        )

    return CestCurveS, x_calcentires


def cest_correction_process(arguments):
    arguments = list(arguments)
    return arguments[-1], calc_pixel(
        arguments[0],
        arguments[1],
        arguments[2],
        arguments[3],
        arguments[4],
        arguments[5],
    )


def calc_pixel(range, y_values, offset, x_itp, x, interpolation):
    y_itp = matlab_style_functions.interpolate(x, y_values, x_itp, interpolation)

    vind_sC_1 = abs(x_itp - (-range + offset))
    vind_sC_2 = abs(x_itp - (range + offset))
    ind_sC_1 = np.argmin(vind_sC_1)
    ind_sC_2 = np.argmin(vind_sC_2)

    y_calcentries = y_itp[ind_sC_1 : ind_sC_2 + 1]
    return y_calcentries
