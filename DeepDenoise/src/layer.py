import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    A module for performing double convolution, consisting of two sequential convolutional layers
    each followed by Batch Normalization and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the double convolution block."""
        return self.double_conv(x)


class Down(nn.Module):
    """
    A module for downsampling the feature map using Max Pooling followed by a double convolution block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the downsampling block."""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    A module for upsampling the feature map either using bilinear upsampling or transposed convolution,
    followed by a double convolution block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bilinear (bool): If True, uses bilinear upsampling, otherwise uses transposed convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else \
            nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass of the upsampling block."""
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    A module representing the output convolution with a 1x1 kernel size.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the output convolution block."""
        return self.conv(x)


class ResDown(nn.Module):
    """
    A module representing a residual down block with two convolutional layers, Batch Normalization, ReLU,
    and a shortcut connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # 1x1 convolutional layer to change the number of channels in the residual
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_shortcut = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual down block."""
        residual = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))

        # Update residual to match the number of channels
        residual = self.bn_shortcut(self.shortcut(residual))
        out += residual
        return out


class ResUp(nn.Module):
    """
    A module representing a residual up block with either bilinear upsampling or transposed convolution,
    followed by a double convolution block and a residual connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bilinear (bool): If True, uses bilinear upsampling, otherwise uses transposed convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else \
            nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

        # Residual connection
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual up block."""
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        # Residual connection
        residual = self.shortcut(x)
        out = self.conv(x) + residual
        return out
