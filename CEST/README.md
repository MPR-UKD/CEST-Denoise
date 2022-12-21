<span style="color:red; font-size:24px">!!! Currently still in progress !!!</span>

# Overview

The **'CEST'** class is a class for performing Chemical Exchange Saturation Transfer (CEST) correction on CEST data. It has three main methods:

- **'__init__'**: This is the constructor for the CEST class. It initializes the class with the following arguments:
- **'cest'**: a numpy array containing the CEST data.
- **'wassr'**: a numpy array containing the Water-Referenced Spectral Saturation Recovery (WASSR) data.
- **'mask'**: a numpy array containing a mask for the CEST data.
- **'cest_range'**: a float indicating the range of the CEST data.
- **'wassr_range'**: a float indicating the range of the WASSR data.
- **'itp_step'**: a float indicating the step size to use for interpolation.
- **'max_wassr_offset'**: a float indicating the maximum offset to use for the WASSR correction.
- **'run'**: This method performs both WASSR correction and CEST correction on the data. It first calls the wassr_correction method, and then calls the cest_correction method and returns the result.

## Example
````python
# Import the numpy library
import numpy as np
from CEST.src.CEST import CEST
from Metrics.src.CEST import mtr_asym
from test_support_function.CEST import generate_Z_3D
import matplotlib.pyplot as plt

# Generate some fake CEST and WASSR data and a mask for the CEST data

# WASSR - 21 dynamics, ppm offsets between -1 and 1, lorentzian amplitude 1, frequency offset 0.5 ppm
wassr = generate_Z_3D(img_size=(10, 10), dyn=21, ppm=1, a=1, b=0.5, c=1)
# Two pool CEST Data with 41 dynamics and a CEST peak at 2 ppm
cest = generate_Z_3D(img_size=(2, 2), dyn=41, ppm=4, a=1, b=0.5, c=1) + generate_Z_3D(img_size=(2, 2), dyn=41, ppm=4, a=0.2, b=2, c=0.5)
    
mask = np.ones((10, 10))

# Create a CEST object with the fake data and some config values
cest_obj = CEST(
    cest=cest,
    wassr=wassr,
    mask=mask,
    cest_range=4,
    wassr_range=1,
    itp_step=0.05,
    max_wassr_offset=1,
)

# Run the CEST correction
corrected_cest, x_calcentires = cest_obj.run()
_, mtr_asym_img = mtr_asym(corrected_cest, mask, (1.5, 2.5), x_calcentires.max())
plt.plot(x_calcentires, corrected_cest[0, 0, :])
print(f'MTRasym: {mtr_asym_img.mean()}')
````

## TODOs:

1. Multi Lorentzian analysis (Module: Metric.CEST)
2. AREX implementation (Module: Metric.CEST)
3. LAREX implementation (Module: Metric.CEST)


## Contribution

This **CEST** project is open source, and anyone is welcome to contribute to it. If you have an idea for a new feature or have found a bug, you can create a pull request or open an issue on the project's GitHub page.

Before making any changes to the code, it is recommended to ensure that the pytests in the **'CEST/test'** directory are still working as expected. This will help to ensure that the changes you are making do not break any existing functionality.

We encourage any and all contributions to this project, and we appreciate your help in making it better for everyone.

# References

1. Lorentzian-Corrected Apparent Exchange-Dependent Relaxation (LAREX) Ω-Plot Analysis-An Adaptation for qCEST in a Multi-Pool System: Comprehensive In Silico, In Situ, and In Vivo Studies (Radke et al., doi: 10.3390/ijms23136920 )
2. Chemical Exchange Saturation Transfer for Lactate-Weighted Imaging at 3 T MRI: Comprehensive In Silico, In Vitro, In Situ, and In Vivo Evaluations (Radke et al., doi: 10.3390/tomography8030106)