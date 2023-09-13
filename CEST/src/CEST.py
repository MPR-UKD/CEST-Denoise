import numpy as np

from CEST.src.WASSR import WASSR
# Import the cest_correction and WASSR functions from the CEST package
from CEST.src.cest_correction import cest_correction


class CEST:
    def __init__(
        self,
        cest: np.ndarray,  # numpy array containing the CEST data
        wassr: np.ndarray,  # numpy array containing the WASSR data
        mask: np.ndarray,  # numpy array containing the mask for the CEST data
        cest_range: float,  # range of the CEST data
        wassr_range: float,  # range of the WASSR data
        itp_step: float,  # step size to use for interpolation
        max_wassr_offset: float,  # maximum offset to use for the WASSR correction
    ):
        self.cest = cest
        self.wassr = wassr
        self.mask = mask
        self.config = {
            "cest_range": cest_range,  # range of the CEST data
            "wassr_range": wassr_range,  # range of the WASSR data
            "itp_step": itp_step,  # step size to use for interpolation
            "max_wassr_offset": max_wassr_offset,  # maximum offset to use for the WASSR correction
        }

    def run(self):
        """Perform both WASSR correction and CEST correction on the data."""
        self.__wassr_correction()  # perform WASSR correction
        return self.__cest_correction()  # perform CEST correction

    def __wassr_correction(self):
        """Perform WASSR correction on the data."""
        # Create a WASSR object with the maximum offset and ppm range specified in the config
        wassr = WASSR(
            max_offset=self.config["max_wassr_offset"], ppm=self.config["wassr_range"]
        )
        # Calculate the offset map and mask for the WASSR data
        self.offset_map, self.mask = wassr.calculate(
            wassr=self.wassr,  # WASSR data
            mask=self.mask,  # mask for the CEST data
            hStep=self.config["itp_step"],  # step size to use for interpolation
        )
        b = 2

    def __cest_correction(self):
        """Perform CEST correction on the data."""
        # Calculate the corrected x-axis values for the CEST data
        x_calcentires = np.arange(
            -self.config["cest_range"] + self.config["max_wassr_offset"],
            self.config["cest_range"] - self.config["max_wassr_offset"],
            self.config["itp_step"],
        )
        x_calcentires = np.append(
            x_calcentires, self.config["cest_range"] - self.config["max_wassr_offset"]
        )
        dyn = self.cest.shape[-1]
        step_size = (self.config["cest_range"] * 2) / (dyn - 1)

        x = np.arange(
            -self.config["cest_range"], self.config["cest_range"], step_size
        ).transpose()
        x = np.append(x, self.config["cest_range"])
        x_itp = np.arange(
            -self.config["cest_range"],
            self.config["cest_range"],
            self.config["itp_step"],
        ).transpose()
        x_itp = np.append(x_itp, self.config["cest_range"])

        mask = self.mask
        self.CestCurveS, self.x_calcentires = cest_correction(
            self.cest,
            x_calcentires,
            x,
            x_itp,
            mask,
            self.offset_map,
            "quadratic",
            self.config["cest_range"] - self.config["max_wassr_offset"],
        )
        return self.CestCurveS, self.x_calcentires
