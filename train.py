import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from DeepDenoise.src.dataloader import CESTDataModule
from DeepDenoise.src.res_unet import CESTResUNet
from DeepDenoise.src.unet import CESTUnet
import torch


def save_onnx_model(model, save_path="model.onnx"):
    """
    Save the PyTorch model in ONNX format.

    Args:
        model (torch.nn.Module): PyTorch model.
        save_path (str): Path to save the ONNX model.
    """
    dummy_input = torch.randn(
        1, model.input_shape[0], model.input_shape[1], model.input_shape[2]
    )
    torch.onnx.export(model, dummy_input, save_path)
    print(f"Model saved in ONNX format at {save_path}")


def main(args: argparse.Namespace) -> None:
    """
    Main function to train and test the model.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    # Instantiate the model
    model_cls = CESTUnet if args.model == "unet" else CESTResUNet
    model = model_cls(
        input_shape=(args.dyn, 128, 128),
        depth=args.depth,
        learning_rate=args.learning_rate,
        noise_estimation=args.noise_estimation,
    )

    # Instantiate the data module
    data_module = CESTDataModule(
        dir=args.data_dir,
        batch_size=args.batch_size,
        workers=args.num_workers,
        noise_std=args.sigma,
        dyn=args.dyn,
    )

    # Define callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    filename = f"{args.model}-lr={args.learning_rate}-noise_estimation={'yes' if args.noise_estimation else 'no'}-best_model"

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.checkpoint_dir) / args.model,
        filename=filename,
        monitor="val_loss",
        mode="min",
        save_top_k=1,  # Save only the best model
    )

    # Instantiate the trainer
    cuda = True if args.gpus != 0 else False
    devices = [_ for _ in range(args.gpus)]

    trainer = pl.Trainer(
        accelerator="cuda" if cuda else "cpu",
        devices=devices if cuda else 1,
        strategy="auto",
        max_epochs=args.max_epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
    )

    # Train and test the model
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
