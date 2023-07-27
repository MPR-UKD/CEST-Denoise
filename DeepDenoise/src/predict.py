import argparse
import nibabel as nib
import torch
from DeepDenoise.src.model import ResUnet


def main(args):
    # Load the model checkpoint
    checkpoint = torch.load(args.checkpoint_path)

    # Instantiate the model
    model = ResUnet.load_from_checkpoint(args.checkpoint_path)

    # Load the input Nifti file
    nifti_data = nib.load(args.input_path)
    nifti_array = nifti_data.get_fdata()

    # Normalize the data
    nifti_array = (nifti_array - nifti_array.mean()) / nifti_array.std()

    # Convert the data to a PyTorch tensor
    input_tensor = torch.from_numpy(nifti_array).unsqueeze(0).float()

    # Make a prediction with the model
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Save the output Nifti file
    output_array = output_tensor.squeeze(0).cpu().numpy()
    output_array = output_array * nifti_array.std() + nifti_array.mean()

    output_data = nib.Nifti1Image(output_array, affine=nifti_data.affine)
    nib.save(output_data, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add command line arguments
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input Nifti file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output Nifti file",
    )

    args = parser.parse_args()

    # Call the main function with the command line arguments
    main(args)
