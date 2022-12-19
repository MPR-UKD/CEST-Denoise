import numpy as np
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
from numba import jit


@jit(nopython=True)
def matlab_style_gauss2D(shape, sigma=1.0):
    if shape == (0, 0):
        return None

    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m: m + 1, -n: n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()

    if sumh != 0:
        h /= sumh

    return h


def interpolate(x, y, points, type):
    interpolations = {
        'cubic': interpolatePChip1D,
        'quadratic': interpolateAkima
    }
    return interpolations[type](x,y,points)


def interpolateAkima(x, y, points):
    """quadratic interpolation"""
    f = Akima1DInterpolator(x, y)
    return f(points)


def interpolatePChip1D(x, y, points):
    """cubic interpolation"""
    f = PchipInterpolator(x, y)
    return f(points)
