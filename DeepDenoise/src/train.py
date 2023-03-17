import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from DeepDenoise.src.model import ResUnet
from DeepDenoise.src.dataset import DenoiseDataset


def main(args):
    # Instantiate the model
    model = ResUnet(
        input_shape=(42, 128, 128),
        depth=args.depth,
        learning_rate=args.learning_rate
    )

    # Instantiate the data module
    data_module = DenoiseDataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Define the learning rate monitor callback
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Define the model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min"
    )

    # Instantiate the trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
        auto_lr_find=True,
        callbacks=[lr_monitor, checkpoint_callback]
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module.test_dataloader())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add command line arguments
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Path to checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--depth", type=int, default=4, help="Depth of the model")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs to train for")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=20, help="Refresh rate of the progress bar")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate")

    args = parser.parse_args()

    # Call the main function with the command line arguments
    main(args)
