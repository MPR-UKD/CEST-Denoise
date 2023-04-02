import torch
from pathlib import Path
import nibabel as nib
from DeepDenoise.src.res_unet import CESTResUNet
from DeepDenoise.src.unet import CESTUnet
import pytest

def test_unet():
    # Initialize the model
    model = CESTUnet(input_shape=(42, 128, 128))

    # Generate a random input tensor
    x = torch.randn((1, 42, 128, 128))

    # Test the forward pass
    y = model(x)

    # Check the output shape
    assert y.shape == (1, 42, 128, 128)

    # Test the model on a real CEST image
    img_path = Path("test/test_data/00001.nii")
    img_nii = nib.load(img_path)
    img = img_nii.get_fdata().transpose(2, 3, 0, 1)
    img_tensor = torch.from_numpy(img).float()

    denoised_img = model(img_tensor).detach().numpy()

    # Check that the denoised image is different from the original image
    assert not (denoised_img == img).all()

    # Check that the denoised image has the correct shape
    assert denoised_img.shape == (1, 42, 128, 128)


def test_res_unet():
    # Initialize the model
    model = CESTResUNet(input_shape=(42, 128, 128))

    # Generate a random input tensor
    x = torch.randn((1, 42, 128, 128))

    # Test the forward pass
    y = model(x)

    # Check the output shape
    assert y.shape == (1, 42, 128, 128)

    # Test the model on a real CEST image
    img_path = Path("test/test_data/00001.nii")
    img_nii = nib.load(img_path)
    img = img_nii.get_fdata().transpose(2, 3, 0, 1)
    img_tensor = torch.from_numpy(img).float()

    denoised_img = model(img_tensor).detach().numpy()

    # Check that the denoised image is different from the original image
    assert not (denoised_img == img).all()

    # Check that the denoised image has the correct shape
    assert denoised_img.shape == (1, 42, 128, 128)


if __name__ == "__main__":
    pytest.main()
