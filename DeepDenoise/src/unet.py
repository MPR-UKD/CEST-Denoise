import torch
import torch.nn as nn
import pytorch_lightning as pl
from DeepDenoise.src.layer import *


class CESTUnet(pl.LightningModule):
    def __init__(
        self,
        input_shape=(42, 128, 128),
        depth: int = 4,
        learning_rate=1e-3,
        noise_estimation: bool = False,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.depth = depth
        self.learning_rate = learning_rate
        self.noise_estimation = noise_estimation

        # Encoder
        self.encoder = nn.ModuleList()
        in_channels = input_shape[0]
        features = 100
        self.inc = DoubleConv(in_channels, features)
        for i in range(depth):
            self.encoder.append(Down(features, features * 2))
            features *= 2

        # Latent space
        self.latent_space = Down(features, features)

        features *= 2
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(depth):
            self.decoder.append(Up(int(features), int(features / 4), True))
            features /= 2
        self.decoder.append(Up(int(features), int(features / 2)))

        self.output_layer = OutConv(int(features / 2), input_shape[0])

        # Define loss function
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # Move input data to the GPU
        x = x.to(self.device)
        input_img = x

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

    def training_step(self, batch, batch_idx):
        x, y = batch["noisy"], batch["ground_truth"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["noisy"], batch["ground_truth"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["noisy"], batch["ground_truth"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
