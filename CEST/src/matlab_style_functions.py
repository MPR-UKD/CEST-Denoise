import numpy as np
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
from numba import jit
from typing import Tuple


# @jit(nopython=True)
# Define a function to generate a 2D Gaussian kernel using the shape and sigma as inputs
def matlab_style_gauss2D(
    shape: Tuple[int, int], sigma: float = 1.0
) -> np.ndarray | None:
    # If the shape is (0, 0), return None
    if shape == (0, 0):
        return None

    # Get the size of the kernel along each dimension
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    # Generate the x and y indices for the kernel
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    # Calculate the values of the kernel using the Gaussian equation
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    # Set all values less than the machine epsilon times the maximum value to zero
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # Normalize the kernel by dividing all values by the sum of the values
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    return h


# Define a function to interpolate using a specified type of interpolation
def interpolate(
    x: np.ndarray, y: np.ndarray, points: np.ndarray, type: str
) -> np.ndarray:
    # Define a dictionary of interpolation functions, where the keys are the strings 'cubic' and 'quadratic'
    interpolations = {"cubic": interpolatePChip1D, "quadratic": interpolateAkima}
    # Return the output of the interpolation function specified by the input 'type'
    return interpolations[type](x, y, points)


# Define a function to perform quadratic interpolation using the Akima1DInterpolator from scipy
def interpolateAkima(x: np.ndarray, y: np.ndarray, points: np.ndarray) -> np.ndarray:
    # Create an Akima1DInterpolator object using the input x and y data
    f = Akima1DInterpolator(x, y)
    # Use the interpolator object to evaluate the points
    return f(points)


# Define a function to perform cubic interpolation using the PchipInterpolator from scipy
def interpolatePChip1D(x: np.ndarray, y: np.ndarray, points: np.ndarray) -> np.ndarray:
    # Create a PchipInterpolator object using the input x and y data
    f = PchipInterpolator(x, y)
    # Use the interpolator object to evaluate the points
    return f(points)
