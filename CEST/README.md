<span style="color:red; font-size:24px">!!! Currently in progress !!!</span>

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
- **'wassr_correction'**: This method performs WASSR correction on the data. It uses the WASSR class to calculate the offset map and mask for the WASSR data, and updates the offset_map and mask attributes of the CEST object with the calculated values.
- **'cest_correction'**: This method performs CEST correction on the data. It interpolates the CEST data using the offset map and mask calculated in the wassr_correction method, and returns the corrected CEST curve and the corrected x-axis values for the CEST data.

The cest_correction function is a separate function from the CEST class, and is used in the cest_correction method of the CEST class to perform the actual CEST correction. It takes the following arguments:

- **'cest'**: a numpy array containing the CEST data.
- **'x_calcentires'**: a numpy array containing the corrected x-axis values for the CEST data.
- **'x'**: a numpy array containing the original x-axis values for the CEST data.
- **'x_itp'**: a numpy array containing the x-axis values to use for interpolation.
- **'mask'**: a numpy array containing a mask for the CEST data.
- **'offset_map'**: a numpy array containing the offset map for the WASSR data.
- **'interp_method'**: a string indicating the interpolation method to use.
- **'x_max'**: a float indicating the maximum x-value to use for the CEST data.

It returns the corrected CEST curve and the corrected x-axis values for the CEST data.