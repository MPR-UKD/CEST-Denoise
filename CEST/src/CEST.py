import numpy as np

from CEST.src.WASSR import WASSR
from CEST.src.cest_correction import cest_correction


class CEST:
    def __init__(
        self,
        cest_data: np.ndarray,
        wassr_data: np.ndarray,
        mask: np.ndarray,
        cest_range: float,
        wassr_range: float,
        interpolation_step: float,
        max_wassr_offset: float,
    ):
        """
        Initialize CEST object.

        Args:
            cest_data (np.ndarray): Array containing the CEST data.
            wassr_data (np.ndarray): Array containing the WASSR data.
            mask (np.ndarray): Array containing the mask for the CEST data.
            cest_range (float): Range of the CEST data.
            wassr_range (float): Range of the WASSR data.
            interpolation_step (float): Step size to use for interpolation.
            max_wassr_offset (float): Maximum offset to use for the WASSR correction.
        """
        self.cest_data = cest_data
        self.wassr_data = wassr_data
        self.mask = mask
        self.config = {
            "cest_range": cest_range,
            "wassr_range": wassr_range,
            "interpolation_step": interpolation_step,
            "max_wassr_offset": max_wassr_offset,
        }

    def run(self) -> tuple:
        """
        Perform both WASSR and CEST correction on the data and return corrected CEST curve and calculated x-axis values.

        Returns:
            tuple: Corrected CEST curve and calculated x-axis values.
        """
        self._perform_wassr_correction()
        return self._perform_cest_correction()

    def _perform_wassr_correction(self):
        """Perform WASSR correction on the data."""
        wassr_instance = WASSR(
            max_offset=self.config["max_wassr_offset"], ppm=self.config["wassr_range"]
        )
        self.offset_map, self.mask = wassr_instance.calculate(
            wassr=self.wassr_data,
            mask=self.mask,
            hStep=self.config["interpolation_step"],
        )

    def _perform_cest_correction(self) -> tuple:
        """Perform CEST correction on the data and return corrected CEST curve and calculated x-axis values."""
        x_calcentires = np.arange(
            -self.config["cest_range"] + self.config["max_wassr_offset"],
            self.config["cest_range"] - self.config["max_wassr_offset"],
            self.config["interpolation_step"],
        )
        x_calcentires = np.append(
            x_calcentires, self.config["cest_range"] - self.config["max_wassr_offset"]
        )

        dyn = self.cest_data.shape[-1]
        step_size = (self.config["cest_range"] * 2) / (dyn - 1)
        x = np.arange(
            -self.config["cest_range"], self.config["cest_range"], step_size
        ).transpose()
        x = np.append(x, self.config["cest_range"])

        x_itp = np.arange(
            -self.config["cest_range"],
            self.config["cest_range"],
            self.config["interpolation_step"],
        ).transpose()
        x_itp = np.append(x_itp, self.config["cest_range"])

        corrected_cest_curve, x_calcentires = cest_correction(
            self.cest_data,
            x_calcentires,
            x,
            x_itp,
            self.mask,
            self.offset_map,
            "quadratic",
            self.config["cest_range"] - self.config["max_wassr_offset"],
        )
        return corrected_cest_curve, x_calcentires
