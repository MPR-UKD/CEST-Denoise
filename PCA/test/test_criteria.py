from PCA.src.denoise import *
from PCA.src.utils import *
from Transform.src.noise import Noiser
from test_support_function.CEST import generate_Z_3D


# Function to generate synthetic CEST data with specified image shape, dynamic range, and noise level
def gen_data(img_shape: tuple, dyn: int, sigma: float):
    # Initialize Noiser object with specified noise level
    n = Noiser(sigma=sigma)
    # Generate CEST data with different chemical shifts
    Z = generate_Z_3D(img_shape, dyn, 3, a=0.5, b=1, c=0.5, delta=0.05) + \
        generate_Z_3D(img_shape, dyn, 3, 1, 0, 0.5, delta=0.01)
    Z2 = generate_Z_3D(img_shape, dyn, 3, a=0.8, b=1, c=0.5, delta=0.05) + \
         generate_Z_3D(img_shape, dyn, 3, 1, 0, 0.5, delta=0.01)
    Z[5:30, 5:22, :] = Z2[5:30, 5:22, :]
    # Add noise to the combined data
    Z_noise = n.add_noise(Z)
    # Return the original data and the data with added noise
    return Z, Z_noise


# Test function for the malinowski_criteria function
def test_criteria():
    # Generate synthetic CEST data with noise level 0
    Z, Z_noise = gen_data((42, 42), 100, 0)
    # Perform steps 1 and 2 of the PCA denoising process on the data
    C_tilde, Z_mean = step1(Z, np.ones((42, 42)))
    eigvals, eigvecs = step2(C_tilde)
    # Call criteria functions and check that the returned value of k is greater than 50
    k_malinowski = malinowski_criteria(eigvals, C_tilde.shape)
    k_median = median_criteria(eigvals)
    k_nelson = nelson_criteria(eigvals, C_tilde.shape)

    assert k_malinowski == int(k_malinowski)
    assert k_median == int(k_median)
    assert k_nelson == int(k_nelson)
    # Generate synthetic CEST data with noise level 0.5
    Z, Z_noise = gen_data((42, 42), 100, 0.1)
    # Perform steps 1 and 2 of the PCA denoising process on the data
    C_tilde, Z_mean = step1(Z, np.ones((42, 42)))
    eigvals, eigvecs = step2(C_tilde)
    # Call criteria functions and check that the returned value of k is lower than 50
    k_malinowski = malinowski_criteria(eigvals, C_tilde.shape)
    k_median = median_criteria(eigvals)
    k_nelson = nelson_criteria(eigvals, C_tilde.shape)
    assert k_malinowski == int(k_malinowski)
    assert k_median == int(k_median)
    assert k_nelson == int(k_nelson)
