import numpy as np

from CEST.cest_correction import cest_correction
from CEST.WASSR import WASSR
import scipy.io as sio


class CEST:
    def __init__(self,
                 cest: np.ndarray,
                 wassr: np.ndarray,
                 mask: np.ndarray,
                 cest_range: float,
                 wassr_range: float,
                 itp_step: float,
                 max_wassr_offset: float):
        self.cest = cest
        self.wassr = wassr
        self.mask = mask
        self.config = {
            'cest_range': cest_range,
            'wassr_range': wassr_range,
            'itp_step': itp_step,
            'max_wassr_offset': max_wassr_offset,
        }

    def run(self):
        self.wassr_correction()
        return self.cest_correction()

    def wassr_correction(self):
        wassr = WASSR(
            max_offset=self.config['max_wassr_offset'],
            ppm=self.config['wassr_range']
        )
        self.offset_map, self.mask = wassr.calculate(wassr=self.wassr,
                                                     mask=self.mask,
                                                     hStep=self.config['itp_step'])

    def cest_correction(self):
        x_calcentires = np.arange(-self.config['cest_range'] + self.config['max_wassr_offset'],
                                  self.config['cest_range'] - self.config['max_wassr_offset'],
                                  self.config['itp_step'])
        x_calcentires = np.append(x_calcentires, self.config['cest_range'] - self.config['max_wassr_offset'])
        dyn = self.cest.shape[-1]
        step_size = (self.config['cest_range'] * 2) / (dyn - 1)

        x = np.arange(-self.config['cest_range'], self.config['cest_range'], step_size).transpose()
        x = np.append(x, self.config['cest_range'])
        x_itp = np.arange(-self.config['cest_range'], self.config['cest_range'], self.config['itp_step']).transpose()
        x_itp = np.append(x_itp, self.config['cest_range'])

        mask = self.mask
        self.CestCurveS, self.x_calcentires = cest_correction(self.cest, x_calcentires, x, x_itp, mask,
                                                              self.offset_map, "quadratic", self.config['cest_range'] -  self.config['max_wassr_offset'])
        return self.CestCurveS, self.x_calcentires
