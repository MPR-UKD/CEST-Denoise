import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from DeepDenoise.src.dataloader import CESTDataModule
from DeepDenoise.src.res_unet import CESTResUNet
from DeepDenoise.src.unet import CESTUnet


def save_onnx_model(model: torch.nn.Module, save_path: str = "model.onnx") -> None:
    """
    Save the PyTorch model in ONNX format.

    Args:
        model (torch.nn.Module): The model to be saved.
        save_path (str, optional): The location where the model should be saved.
    """
    # Create dummy input tensor with the same input shape as the model expects
    dummy_input = torch.randn(
        1, model.input_shape[0], model.input_shape[1], model.input_shape[2]
    )

    # Export model in ONNX format
    torch.onnx.export(model, dummy_input, save_path)
    print(f"Model saved in ONNX format at {save_path}")


def main(args: argparse.Namespace) -> None:
    """
    Main function to train and test the model.

    Args:
        args (argparse.Namespace): Parsed command line arguments containing model training parameters.
    """
    # Select the model class based on the chosen model type
    model_cls = CESTUnet if args.model == "unet" else CESTResUNet

    # Instantiate the model with the provided parameters
    model = model_cls(
        input_shape=(args.dyn, 128, 128),
        depth=args.depth,
        learning_rate=args.learning_rate,
        noise_estimation=args.noise_estimation,
    )

    # Instantiate the data module with the provided parameters
    data_module = CESTDataModule(
        dir=args.data_dir,
        batch_size=args.batch_size,
        workers=args.num_workers,
        noise_std=args.sigma,
        dyn=args.dyn,
    )

    # Define callbacks for learning rate monitoring and model checkpointing
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Filename for saving the best model's checkpoint
    filename = f"{args.model}-lr={args.learning_rate}-noise_estimation={'yes' if args.noise_estimation else 'no'}-best_model"
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.checkpoint_dir) / args.model,
        filename=filename,
        monitor="val_loss",
        mode="min",
        save_top_k=1,  # Save only the best model
    )

    # Define the device and strategy for model training
    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus > 0 else "cpu",
        gpus=args.gpus if args.gpus > 0 else None,
        max_epochs=args.max_epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
    )

    # Train and test the model using the trainer
    trainer.fit(model, data_module)
    trainer.test(model, data_module.test_dataloader())

    # Save the best model in ONNX format
    best_model_path = checkpoint_callback.best_model_path
    best_model = model_cls.load_from_checkpoint(
        best_model_path,
        input_shape=(args.dyn, 128, 128),
        depth=args.depth,
        learning_rate=args.learning_rate,
        noise_estimation=args.noise_estimation,
    )

    # Save the optimized model
    save_onnx_model(
        best_model, Path(args.checkpoint_dir) / args.model / (filename + ".onnx")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test the CEST model.")

    # Add command line arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="DeepDenoise/test/test_data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for data loading"
    )
    parser.add_argument("--depth", type=int, default=3, help="Depth of the model")
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Maximum number of epochs to train for",
    )
    parser.add_argument(
        "--dyn",
        type=int,
        default=41,
        help="Number of offset frequencies in the Z-spectrum",
    )
    parser.add_argument(
        "--gpus", type=int, default=0, help="Number of GPUs to use for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Initial learning rate"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["unet", "resunet"],
        default="unet",
        help="Model type",
    )
    parser.add_argument(
        "--noise_estimation", action="store_true", help="Enable noise estimation"
    )
    parser.add_argument(
        "--sigma", type=float, default=0.1, help="Standard deviation of the noise"
    )

    args = parser.parse_args()

    # Call the main function with the command line arguments
    main(args)
