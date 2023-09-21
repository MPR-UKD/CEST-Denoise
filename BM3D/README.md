# BM3D Image Denoising Library

BM3D is a comprehensive library providing implementations for denoising 2D and 3D images using the Block-Matching and 3D Filtering (BM3D) algorithm. It offers easy-to-use functions and customization options, allowing users to deal with different kinds of noise in images effectively.

## Key Features
- **Efficient Denoising**: Offers advanced denoising techniques for both 2D and 3D images.
- **Custom Configurations**: Allows users to specify various parameters to tailor the denoising process to specific needs.
- **Multi-Processing Support**: Provides options for utilizing multiple processors to expedite the denoising process, especially useful for large datasets.
- **Versatile Application**: Includes specialized functions for denoising Chemical Exchange Saturation Transfer (CEST) data.

## How to Use

### Main Function
The principal function, `bm3d(img, config=None, mask=None)`, executes the denoising process with the following parameters:
- **img**: The target image represented as a 2D or 3D numpy array.
- **config**: A dictionary containing user-defined configuration parameters. Defaults will be applied if not provided.
- **mask**: An optional mask defining the pixels to undergo denoising.

### BM3D for CEST Data
For CEST data, the function `bm3d_CEST(img, mask=None, config=None, multi_processing=False)` is available. It works similarly to the main function but with an additional `multi_processing` parameter to enable or disable parallel processing.

### Configuration Parameters
Users can customize the denoising process through the following parameters in the `config` dictionary:
- **'sigma'**: Defines the standard deviation of the Gaussian noise in the image.
- **'lamb2d'** & **'lamb3d'**: Control the strength of the 2D and 3D transformations in the denoising steps.
- **'KaiserWindowBeta'**: Determines the shape of the Kaiser window used in block matching.
- **'TransformationFunction'**: Specifies the discrete transformation function, either "cos" or "sin".
- **'step1_threshold_distance'** & **'step2_threshold_distance'**: Threshold distances for the first and second denoising steps.
- **'step1_max_match'** & **'step2_max_match'**: Maximum number of matched blocks in each denoising step.
- **'step1_BlockSize'** & **'step2_BlockSize'**: Define the block sizes for the respective denoising steps.
- **'step1_WindowSize'** & **'step2_WindowSize'**: Set the search window sizes for each step.

### Example Usage
```python
import numpy as np
from BM3D.src.denoising import bm3d

# Load a noisy image as a numpy array
noisy_image = np.load('noisy_image.npy')

# Specify the configuration parameters
config = {'sigma': 25, 'lamb2d': 2, 'lamb3d': 2.7, 'KaiserWindowBeta': 2, 'TransformationFunction': 'cos'}

# Perform denoising
denoised_image = bm3d(noisy_image, config)

# Save the resultant image
np.save('denoised_image.npy', denoised_image)
```

## Performance Considerations
BM3D is resource-intensive, especially for large images. To optimize performance, users are encouraged to experiment with different configurations to strike a balance between denoising quality and computational efficiency.

## Contributing to BM3D
BM3D is an open-source project. Contributions, whether in the form of new features, bug fixes, or other improvements, are welcome. Please ensure that any changes made pass the tests in the **'BM3D/test'** directory to maintain the integrity of the library.

Visit [BM3D GitHub page](#) to open issues or submit pull requests.

## Acknowledgments
We are grateful to the contributors and users of BM3D, who help in refining the library and making it a valuable tool for the community.
