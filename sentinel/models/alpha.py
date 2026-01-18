"""Spatial Alpha Controller with 3x3 convolutions for corruption detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAlphaController(nn.Module):
    """
    Spatial alpha controller using 3x3 convolutions for neighborhood context.

    This is the FIXED version that replaces the broken 1x1 conv AlphaController2D.

    Architecture:
    - Input: 2 channels (resid_mag + redundancy)
    - Hidden: 3 layers of 3x3 convs with GroupNorm and SiLU
    - Output: 1 channel alpha map, clamped to [alpha_min, alpha_max]
    - Fully convolutional: handles any input size without retraining

    Effective receptive field: 7x7 (3 layers of 3x3)
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 32,
        num_layers: int = 3,
        alpha_min: float = -1.5,
        alpha_max: float = 1.9,
        alpha0: float = -0.3
    ):
        """
        Args:
            in_channels: Input channels (default 2: resid_mag + redundancy)
            hidden_channels: Hidden layer capacity (default 32)
            num_layers: Number of conv layers (default 3 for 7x7 receptive field)
            alpha_min: Minimum alpha value (default -1.5, robust loss)
            alpha_max: Maximum alpha value (default 1.9, L2-like loss)
            alpha0: Bias offset (default -0.3)
        """
        super().__init__()

        layers = []

        # First layer: in_channels -> hidden_channels
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.GroupNorm(min(8, hidden_channels), hidden_channels))
        layers.append(nn.SiLU())

        # Middle layers: hidden -> hidden
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(min(8, hidden_channels), hidden_channels))
            layers.append(nn.SiLU())

        # Final layer: hidden -> 1 (no activation, raw logits)
        layers.append(nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha0 = alpha0

        # Initialize final layer to output near alpha0
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, resid_mag: torch.Tensor, redundancy: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            resid_mag: (B, 1, H, W) - Reconstruction residual magnitude
            redundancy: (B, 1, H, W) - Local redundancy score

        Returns:
            alpha: (B, 1, H, W) - Per-pixel alpha values in [alpha_min, alpha_max]
        """
        x = torch.cat([resid_mag, redundancy], dim=1)  # (B, 2, H, W)
        a = self.net(x) + self.alpha0
        return torch.clamp(a, self.alpha_min, self.alpha_max)


def local_redundancy(x: torch.Tensor, window: int = 5) -> torch.Tensor:
    """
    Compute local redundancy (normalized local variance).

    High redundancy = low variance = predictable/smooth regions
    Low redundancy = high variance = edges/textures

    Args:
        x: Input tensor (B, C, H, W), typically grayscale (C=1)
        window: Window size for local statistics (default 5)

    Returns:
        redundancy: (B, C, H, W) - Redundancy score in [0, 1]
    """
    pad = window // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")

    # Local mean and variance
    mean = F.avg_pool2d(x_pad, kernel_size=window, stride=1)
    mean2 = F.avg_pool2d(x_pad * x_pad, kernel_size=window, stride=1)
    var = (mean2 - mean * mean).clamp_min(0.0)

    # Normalize variance to [0, 1] per image
    vmin = var.amin(dim=(2, 3), keepdim=True)
    vmax = var.amax(dim=(2, 3), keepdim=True)
    norm = (var - vmin) / (vmax - vmin + 1e-6)

    # Redundancy = 1 - normalized_variance
    redundancy = 1.0 - norm
    return redundancy
