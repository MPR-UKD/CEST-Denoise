# DeepDenoise: CEST MRI Denoising Framework

## Introduction
DeepDenoise is a Python framework built on PyTorch and PyTorch Lightning designed for denoising CEST MRI data. It includes implementations of U-Net and ResU-Net architectures, data loading utilities, and training scripts, allowing users to train custom models on their datasets.

## Requirements
- Python 3.6 or above
- PyTorch 1.7.0 or above
- PyTorch Lightning
- Nibabel


## Architecture
DeepDenoise contains implementations of two architectures:
1. **U-Net (CESTUnet)**
2. **ResU-Net (CESTResUNet)**

Both architectures are designed for image segmentation tasks but are adapted here for denoising CEST MRI data.

## Dataset
The dataset should consist of `.nii` files placed in a specified directory. The `CESTDataset` class is responsible for loading, processing, and splitting the dataset into training, validation, and testing sets based on the provided distribution.

## Training
To train a model, run the training script with the appropriate arguments:
```shell
python train.py --data_dir <path-to-data> --model unet --max_epochs 100
```

### Arguments
- `--data_dir`: Path to the directory containing the dataset.
- `--model`: Type of model to train (`unet` or `resunet`).
- `--max_epochs`: Maximum number of epochs for training.
- Additional arguments are available; refer to the training script for details.

## Configuration
The framework provides various configuration options, such as batch size, number of workers, learning rate, noise standard deviation, and more, allowing users to tailor the training process to their needs.

## Callbacks
DeepDenoise utilizes PyTorch Lightning callbacks for learning rate monitoring and model checkpointing, ensuring the best model is saved during training.

## Testing
Once the model is trained, the framework evaluates it on the test dataset and logs the results.

## Model Export
The trained model can be exported in ONNX format, enabling deployment in various environments.

## Contributing
Contributions are welcome! Please submit pull requests for any enhancements, bug fixes, or new features.

## Conclusion
DeepDenoise provides a comprehensive solution for denoising CEST MRI data, offering customizable training options, advanced architectures, and seamless integration with PyTorch ecosystems. Whether you are a researcher, a developer, or a healthcare professional, DeepDenoise can aid in enhancing the quality of CEST MRI data, contributing to advancements in medical imaging research and applications.