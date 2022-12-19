import itertools

import numpy as np


def lorentzian(a: float, b: float, c: float, x: np.ndarray) -> np.ndarray:
    """
    Calculate the value of a Lorentzian function at a given x.

    Parameters:
    - a: float, representing the height of the Lorentzian curve
    - b: float, representing the position of the peak of the Lorentzian curve
    - c: float, representing the width of the Lorentzian curve at half-height
    - x: np.ndarray, representing the ppms

    Returns:
    - np.ndarray, representing the values of the Lorentzian function
    """
    return a / (1 + ((x - b) / c) ** 2)


def z_spectra(a: float, b: float, c: float, x: np.ndarray) -> np.ndarray:
    """
    Calculate the value of a Lorentzian function at a given x.

    Parameters:
    - a: float, representing the height of the Lorentzian curve
    - b: float, representing the position of the peak of the Lorentzian curve
    - c: float, representing the width of the Lorentzian curve at half-height
    - x: np.ndarray, representing the ppms

    Returns:
    - np.ndarray, representing the values of the Lorentzian function
    """
    return 1 - lorentzian(a, b, c, x)


def generate_Z_3D(
    img_size: tuple, dyn: int, ppm: float, a: float = 0.1, b: float = 1, c: float = 3
) -> np.ndarray:
    """
    Generate a 3D array of Z values.

    Parameters:
    - img_size: tuple of int, representing the size of the image (number of rows and columns)
    - dyn: int, representing the number of dynamics in the image
    - ppm: float, representing the range of ppm values in the image
    - a: float, representing the height of the Lorentzian curve (default value is 0.1)
    - b: float, representing the position of the peak of the Lorentzian curve (default value is 1)
    - c: float, representing the width of the Lorentzian curve at half-height (default value is 3)

    Returns:
    - np.ndarray, representing the 3D array of Z values
    """
    Z = np.zeros((img_size[0], img_size[1], dyn))

    step_size = (2 * ppm) / (dyn - 1)
    x = np.arange(-ppm, ppm + 0.001, step_size)

    # Calculate Lorentzian curve
    z = z_spectra(a, b, c, x)

    for x, y in itertools.product(range(img_size[0]), range(img_size[1])):
        Z[x, y, :] = z
    return Z
