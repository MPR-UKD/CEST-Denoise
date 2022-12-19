import numpy as np
from CEST.src.CEST import matlab_style_functions


class WASSR:

    def __init__(self, max_offset, ppm):
        self.max_offset = max_offset
        self.ppm = ppm
        self.offset_map = None

    def calculate(self, wassr, mask, hStep):
        (rows, colums, dyn) = wassr.shape
        OF = np.zeros((rows, colums))

        ppms = np.linspace(-self.ppm, self.ppm, dyn)
        pixels = np.argwhere(mask != 0)

        for i, tuple in enumerate(pixels):
            values = wassr[tuple[0], tuple[1], :]
            OF[tuple[0], tuple[1]] = calc_offset(ppms, values, self.max_offset, hStep, self.ppm)
        self.offset_map = OF
        mask[self.offset_map == -100] = 0
        return self.offset_map, mask


def calc_offset(ppms, values, max_shift, hStep, max_offset):
    if abs(ppms[np.argmin(values)]) > max_shift:
        return -100
    dppm = max_shift
    x_start = np.min(ppms)
    x_end = np.max(ppms)

    x_interp = np.arange(x_start, x_end, hStep).transpose()
    x_interp = np.append(x_interp, x_end)
    x_interp_mirror = -x_interp

    y_interp = matlab_style_functions.interpolate(ppms, values, x_interp, "quadratic")
    minind = np.argmin(y_interp)

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

    OF = msa(max_shift, max_offset, hStep, x_interp_neu, y_interp_neu, x_interp_mirror, y_interp)
    # OF = msa2(x_interp_neu, y_interp_neu, x_interp_mirror, y_interp, self.maxShift, self.maxOffset, self.hStep)
    if abs(OF) > max_shift:
        return -100
    return OF


def msa(maxShift, maxOffset, hStep, xWerte, yWerte, x_interp_mirror, y_interp):
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

        MSE_Vektor = (Ywert_verschobenmirror - yWerte) * (Ywert_verschobenmirror - yWerte)
        MSCF[i] = MSE_Vektor.sum()

    indexmin = np.argmin(MSCF)
    return AbtastvektorC[indexmin]
