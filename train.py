import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from DeepDenoise.src.res_unet import CESTResUNet
from DeepDenoise.src.unet import CESTUnet
from DeepDenoise.src.dataloader import CESTDataModule
from pathlib import Path
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme


def main(args):
    # Instantiate the model
    if args.model == "unet":
        model = CESTUnet(
            input_shape=(42, 128, 128),
            depth=args.depth,
            learning_rate=args.learning_rate,
            noise_estimation=args.noise_estimation,
        )
    else:
        model = CESTResUNet(
            input_shape=(42, 128, 128),
            depth=args.depth,
            learning_rate=args.learning_rate,
            noise_estimation=args.noise_estimation,
        )

    # Instantiate the data module
    data_module = CESTDataModule(
        dir=args.data_dir, batch_size=args.batch_size, workers=args.num_workers
    )

    # Define the learning rate monitor callback
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Define the model checkpoint callback
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

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module.test_dataloader())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        "--gpus", type=int, default=0, help="Number of GPUs to use for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Initial learning rate"
    )
    parser.add_argument("--model", type=str, default="res_unet", help="Model type")
    parser.add_argument(
        "--noise_estimation", type=bool, default=False, help="Noise Estimation"
    )

    args = parser.parse_args()

    # Call the main function with the command line arguments
    main(args)
