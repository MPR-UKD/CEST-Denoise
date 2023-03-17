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


class LorentzianPool:
    def __init__(
            self,
            name: str,
            amplitude_bounds: tuple[float, float, float],
            position_bounds: tuple[float, float, float],
            width_bounds: tuple[float, float, float],
    ):
        self.name = name
        self.amplitude_bounds = amplitude_bounds
        self.position_bounds = position_bounds
        self.width_bounds = width_bounds


def multi_lorentzian(x, *params):
    """
    Multi-Lorentzian function.

    Parameters:
    - x: numpy array, the ppm values
    - params: list of floats, the Lorentzian parameters (A, x0, gamma) for each pool

    Returns:
    - y: numpy array, the Multi-Lorentzian values
    """
    y = np.zeros_like(x)
    num_pools = len(params) // 3

    for i in range(num_pools):
        A, x0, gamma = params[i * 3: (i + 1) * 3]
        y += A * gamma ** 2 / ((x - x0) ** 2 + gamma ** 2)

    y += params[-1]
    return y


def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the coefficient of determination (R^2) for the given true and predicted values.

    Parameters:
    - y_true: numpy array, the true values
    - y_pred: numpy array, the predicted values

    Returns:
    - R^2 value, a float
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def calculate_adjusted_r_squared(y_true: np.ndarray, y_pred: np.ndarray, num_params: int) -> float:
    """
    Calculate the adjusted coefficient of determination (adjusted R^2) for the given true and predicted values.

    Parameters:
    - y_true: numpy array, the true values
    - y_pred: numpy array, the predicted values
    - num_params: int, the number of parameters in the model

    Returns:
    - Adjusted R^2 value, a float
    """
    n = len(y_true)
    r_squared = calculate_r_squared(y_true, y_pred)
    adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - num_params - 1))
    return adjusted_r_squared
