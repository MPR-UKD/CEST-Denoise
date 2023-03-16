import numpy as np

from CEST.src.CEST import CEST
from test_support_function.src.CEST import generate_Z_3D
import pytest
from Metrics.src.CEST import mtr_asym


def test_OF_0():
    # Test CEST with and water offset of 0 ppm
    wassr_img = generate_Z_3D(img_size=(2, 2), dyn=21, ppm=1, a=1, b=0, c=1)
    cest_img = generate_Z_3D(
        img_size=(2, 2), dyn=41, ppm=4, a=1, b=0, c=1
    ) + generate_Z_3D(img_size=(2, 2), dyn=41, ppm=4, a=0.2, b=2, c=0.5)
    cest_img -= cest_img.min()
    cest = CEST(
        cest=cest_img,
        wassr=wassr_img,
        mask=np.ones((2, 2)),
        cest_range=4,
        wassr_range=1,
        itp_step=0.05,
        max_wassr_offset=1,
    )
    CestCurveS, x_calcentires = cest.run()
    _, mtr_asym_img = mtr_asym(
        CestCurveS, np.ones((2, 2)), (1.5, 2.5), x_calcentires.max()
    )
    assert mtr_asym_img[0, 0] == mtr_asym_img[0, 1]
    assert mtr_asym_img[0, 0] > 0
    assert np.argmin(CestCurveS[0, 0]) == np.argmin(abs(x_calcentires))


def test_OF_0_5():
    # Test CEST with and water offset of 0.5 ppm
    wassr_img = generate_Z_3D(img_size=(2, 2), dyn=21, ppm=1, a=1, b=0.5, c=1)
    cest_img = generate_Z_3D(
        img_size=(2, 2), dyn=41, ppm=4, a=1, b=0.5, c=1
    ) + generate_Z_3D(img_size=(2, 2), dyn=41, ppm=4, a=0.2, b=2, c=0.5)
    cest_img -= cest_img.min()
    cest = CEST(
        cest=cest_img,
        wassr=wassr_img,
        mask=np.ones((2, 2)),
        cest_range=4,
        wassr_range=1,
        itp_step=0.05,
        max_wassr_offset=1,
    )
    CestCurveS, x_calcentires = cest.run()
    _, mtr_asym_img = mtr_asym(
        CestCurveS, np.ones((2, 2)), (1.5, 2.5), x_calcentires.max()
    )
    assert mtr_asym_img[0, 0] == mtr_asym_img[0, 1]
    assert mtr_asym_img[0, 0] > 0
    assert np.argmin(CestCurveS[0, 0]) == np.argmin(abs(x_calcentires))


if __name__ == "__main__":
    pytest.main()
