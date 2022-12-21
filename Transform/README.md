# Transform

This repository contains a package for applying various transforms to data. Currently, the package includes a Noiser class for adding white noise to data.

## Noiser

Noiser is a Python class that adds white noise to a given image or a set of images. The noise is added in the k-space of the image, and the resulting image is obtained by transforming the k-space back to the image space.

#### Usage

To use Noiser, you need to create an instance of the Noiser class:

````python
from Transform.src.noise import Noiser

noiser = Noiser(sigma=0.01)
````

The **'sigma'** parameter determines the strength of the noise that will be added to the image. You can change this value at any time by calling the **'set_sigma'** method:

````python
noiser.set_sigma(0.1)
````

To add noise to an 2D image, a set of 2D images, a 3D image or 2D CEST data, you can use the add_noise method:

````python
noisy_imgs = noiser.add_noise(imgs)
````

The **'add_noise'** method takes an **'imgs'** parameter, which should be a NumPy array. The method returns a NumPy array with the same shape as the input array, but with the noise added.

You can also specify a **'sigma'** value when calling the **'add_noise'** method, which will override the sigma value of the Noiser object for this particular call:

````python
noisy_imgs = noiser.add_noise(imgs, sigma=0.05)
````

#### Private Methods

The Noiser class also has some private methods that are used internally to generate and apply the noise:

- **__add_gaussian_noise**: adds Gaussian noise to a k-space image
- **__transform_kspace_to_image**: transforms a k-space image back to the image space
- **__transform_image_to_kspace**: transforms an image to its k-space representation
These methods should not be called directly, and are only intended for internal use by the Noiser class.