import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from DeepDenoise.src.dataloader import CESTDataModule
from DeepDenoise.src.res_unet import CESTResUNet
from DeepDenoise.src.unet import CESTUnet


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
    )

    # Define callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.checkpoint_dir) / args.model,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
    )

    # Instantiate the trainer
    cuda = True if args.gpus != 0 else False
    devices = [_ for _ in range(args.gpus)]

    trainer = pl.Trainer(
        accelerator="cuda" if cuda else "cpu",
        devices=devices if cuda else 1,
        strategy="dp" if cuda else "ddp",
        max_epochs=args.max_epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
    )

    # Train and test the model
    trainer.fit(model, data_module)
    trainer.test(model, data_module.test_dataloader())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test the CEST model.")

    # Add command line arguments
    parser.add_argument("--data_dir", type=str, default="DeepDenoise/test/test_data", help="Path to data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Path to checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
    parser.add_argument("--depth", type=int, default=3, help="Depth of the model")
    parser.add_argument("--max_epochs", type=int, default=2, help="Maximum number of epochs to train for")
    parser.add_argument("--dyn", type=int, default=42, help="Number of offset frequencies in the Z-spectrum")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--model", type=str, choices=["unet", "resunet"], default="unet", help="Model type")
    parser.add_argument("--noise_estimation", action="store_true", help="Enable noise estimation")
    parser.add_argument("--sigma", type=float, default=0.05, help="Standard deviation of the noise")

    args = parser.parse_args()

    # Call the main function with the command line arguments
    main(args)
