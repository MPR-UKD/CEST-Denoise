from typing import Tuple, Dict

import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from DeepDenoise.src.layer import ResDown, ResUp, OutConv
from Metrics.src.image_quality_estimation import check_performance


class CESTResUNet(pl.LightningModule):
    """
    Residual U-Net (ResUNet) architecture for CEST MRI denoising.
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (42, 128, 128),
                 depth: int = 4,
                 learning_rate: float = 1e-3,
                 noise_estimation: bool = False) -> None:
        """
        Args:
            input_shape (Tuple[int, int, int]): Shape of the input data (default is (42, 128, 128)).
            depth (int): Depth of the U-Net (default is 4).
            learning_rate (float): Learning rate for the optimizer (default is 1e-3).
            noise_estimation (bool): If True, the network estimates noise (default is False).
        """
        super().__init__()

        self.input_shape = input_shape
        self.depth = depth
        self.learning_rate = learning_rate
        self.noise_estimation = noise_estimation

        # Encoder
        self.encoder = nn.ModuleList()
        in_channels = input_shape[0]
        features = 100
        self.inc = nn.Sequential(ResDown(in_channels, features), ResDown(features, features))
        for _ in range(depth):
            self.encoder.append(
                nn.Sequential(
                    ResDown(features, features * 2),
                    ResDown(features * 2, features * 2),
                    nn.MaxPool2d(2),
                )
            )
            features *= 2

        # Latent space
        self.latent_space = nn.Sequential(ResDown(features, features), ResDown(features, features))

        # Decoder
        self.decoder = nn.ModuleList()
        for _ in range(depth):
            self.decoder.append(ResUp(features, features // 4, True))
            features //= 2
        self.decoder.append(ResUp(features, features // 2, False))

        self.output_layer = OutConv(features // 2, input_shape[0])

        # Define loss function
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResU-Net.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.to(self.device)
        input_img = x  # Keep a reference to the input image

        # Encoder
        x = self.inc(x)
        encoding_outputs = [x]
        for enc in self.encoder:
            x = enc(x)
            encoding_outputs.append(x)

        # Latent space
        x = self.latent_space(x)

        # Decoder
        for i, dec in enumerate(self.decoder):
            x = dec(x, encoding_outputs[-(i + 1)])

        x = self.output_layer(x)

        return input_img - x if self.noise_estimation else x

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Performs a training step.

        Args:
            batch (Dict[str, torch.Tensor]): Batch data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        x, y = batch["noisy"], batch["ground_truth"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("loss", loss, sync_dist=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Performs a validation step.

        Args:
            batch (Dict[str, torch.Tensor]): Batch data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        x, y = batch["noisy"], batch["ground_truth"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        performance = check_performance(y, x, y_hat)
        self.log_metrics("val", loss, performance)
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Performs a test step.

        Args:
            batch (Dict[str, torch.Tensor]): Batch data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        x, y = batch["noisy"], batch["ground_truth"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        performance = check_performance(y, x, y_hat)
        self.log_metrics("test", loss, performance)
        return loss

    def log_metrics(self, stage: str, loss: torch.Tensor, performance: Dict[str, float]) -> None:
        """
        Logs metrics for the specified stage.

        Args:
            stage (str): The stage for which to log metrics (e.g., "val" or "test").
            loss (torch.Tensor): The loss tensor.
            performance (Dict[str, float]): Dictionary with performance metrics.
        """
        self.log(f"{stage}_loss", loss, sync_dist=True, on_epoch=True, prog_bar=True)
        for metric, value in performance.items():
            self.log(f"{stage}_PSNR_{metric.lower()}", value, sync_dist=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict:
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            Dict: Dictionary with optimizer and lr_scheduler.
        """
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
