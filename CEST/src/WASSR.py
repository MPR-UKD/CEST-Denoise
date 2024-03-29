from typing import Tuple

import numpy as np

from CEST.src import matlab_style_functions


# Water saturation shift referencing
class WASSR:
    # Initialize the class with the maximum shift, ppm range, and initial values for the offset map and mask
    def __init__(self, max_offset: float, ppm: float):
        self.max_offset = max_offset
        self.ppm = ppm
        self.offset_map = None

    def calculate(
        self, wassr: np.ndarray, mask: np.ndarray, hStep: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Get the shape of the WASSR data
        (rows, colums, dyn) = wassr.shape
        OF = np.zeros((rows, colums))

        # Generate a list of ppm values from -ppm to ppm
        ppms = np.linspace(-self.ppm, self.ppm, dyn)
        # Get the indices of the non-zero elements in the mask
        pixels = np.argwhere(mask != 0)

        # Calculate the offset for each non-zero element in the mask
        for i, tuple in enumerate(pixels):
            values = wassr[tuple[0], tuple[1], :]
            OF[tuple[0], tuple[1]] = calc_offset(
                ppms, values, self.max_offset, hStep, self.ppm
            )
        self.offset_map = OF
        # Set all elements in the mask where the offset is -100 to 0
        mask[self.offset_map == -100] = 0
        return self.offset_map, mask


# Calculate the offset for a given set of ppm values and corresponding intensities
def calc_offset(
    ppms: np.ndarray,
    values: np.ndarray,
    max_shift: float,
    hStep: float,
    max_offset: float,
) -> float:
    # If the ppm value with the minimum intensity is greater than the maximum shift, return -100
    if abs(ppms[np.argmin(values)]) > max_shift:
        pass  # return -100
    dppm = max_shift
    x_start = np.min(ppms)
    x_end = np.max(ppms)

    # Generate a list of interpolation points between the start and end ppm values
    x_interp = np.arange(x_start, x_end, hStep).transpose()
    x_interp = np.append(x_interp, x_end)
    # Generate a list of interpolation points mirrored about the y-axis
    x_interp_mirror = -x_interp

    # Interpolate the values using quadratic interpolation
    y_interp = matlab_style_functions.interpolate(ppms, values, x_interp, "quadratic")
    minind = np.argmin(y_interp)

    # Round the ppm value with the minimum intensity to the nearest hundredth
    xsuch = round(x_interp[minind] * 100) / 100
    xsuch_minus = xsuch - dppm
    xsuch_plus = xsuch + dppm

    if xsuch_minus <= x_start:
        xsuch_minus = x_start

    if xsuch_plus >= x_end:
        xsuch_plus = x_end

    x_interp_neu = np.arange(xsuch_minus, xsuch_plus, hStep).transpose()
    x_interp_neu = np.append(x_interp_neu, xsuch_plus)

    y_interp_neu = matlab_style_functions.interpolatePChip1D(ppms, values, x_interp_neu)

    OF = msa(
        max_shift,
        max_offset,
        hStep,
        x_interp_neu,
        y_interp_neu,
        x_interp_mirror,
        y_interp,
    )
    if abs(OF) > max_shift:
        return -100
    return OF


def msa(
    maxShift: float,
    maxOffset: float,
    hStep: float,
    xWerte: np.ndarray,
    yWerte: np.ndarray,
    x_interp_mirror: np.ndarray,
    y_interp: np.ndarray,
) -> float:
    n_points = len(xWerte)
    minind = np.argmin(yWerte)
    x_search = round(xWerte[minind] * 100) / 100

    start_Abt = x_search - maxShift
    if start_Abt <= -maxOffset:
        start_Abt = -maxOffset

    ende_Abt = x_search + maxShift
    if ende_Abt >= maxOffset:
        ende_Abt = maxOffset

    AbtastvektorC = np.arange(start_Abt, ende_Abt, hStep).transpose()
    AbtastvektorC = np.append(AbtastvektorC, ende_Abt)
    siyAC = len(AbtastvektorC)

    MSCF = np.zeros((siyAC), dtype=float)

    for i in range(0, siyAC):
        C = AbtastvektorC[i]
        Xwert_verschobenmirror = np.zeros((n_points), dtype=float)
        Ywert_verschobenmirror = np.zeros((n_points), dtype=float)

        for j in range(0, n_points):
            xn = xWerte[j]
            x_interp_mirror_versch = x_interp_mirror + 2 * C
            V_x_interp_mirror_versch = abs(x_interp_mirror_versch - xn)
            index = np.argmin(V_x_interp_mirror_versch)
            Xwert_verschobenmirror[j] = x_interp_mirror_versch[index]
            Ywert_verschobenmirror[j] = y_interp[index]

        MSE_Vektor = (Ywert_verschobenmirror - yWerte) * (
            Ywert_verschobenmirror - yWerte
        )
        MSCF[i] = MSE_Vektor.sum()

    indexmin = np.argmin(MSCF)
    return AbtastvektorC[indexmin]
