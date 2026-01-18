"""Core QualityGate engine for Sentinel."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

from .models import UNet2D, SpatialAlphaController, local_redundancy


@dataclass
class CheckResult:
    """Result from checking a single image."""
    score: float
    flagged: bool
    alpha_map: Optional[torch.Tensor] = None


class QualityGate:
    """
    Main quality gate engine for detecting corrupted images.

    Usage:
        gate = QualityGate.load(weights_dir)
        results = gate.check_batch(images)
    """

    def __init__(
        self,
        unet: UNet2D,
        alpha_controller: SpatialAlphaController,
        threshold: float = 0.05,
        redundancy_window: int = 5,
        device: str = "auto",
    ):
        """
        Args:
            unet: UNet reconstruction model
            alpha_controller: SpatialAlphaController for alpha maps
            threshold: Score threshold for flagging (default 0.05 = 5% FPR)
            redundancy_window: Window size for local redundancy computation
            device: Device to run on ("cuda", "cpu", "mps", or "auto")
        """
        self.unet = unet
        self.alpha_controller = alpha_controller
        self.threshold = threshold
        self.redundancy_window = redundancy_window

        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.unet.to(self.device)
        self.alpha_controller.to(self.device)

        self.unet.eval()
        self.alpha_controller.eval()

    @classmethod
    def load(
        cls,
        weights_dir: Union[str, Path],
        threshold: float = 0.05,
        device: str = "auto",
    ) -> "QualityGate":
        """
        Load a trained QualityGate from weights directory.

        Args:
            weights_dir: Directory containing unet.pth and alpha_controller.pth
            threshold: Score threshold for flagging
            device: Device to run on

        Returns:
            Loaded QualityGate instance
        """
        weights_dir = Path(weights_dir)

        # Initialize models
        unet = UNet2D(in_ch=3, base=64)
        alpha_controller = SpatialAlphaController(
            in_channels=2,
            hidden_channels=32,
            num_layers=3,
        )

        # Load weights
        unet_path = weights_dir / "unet.pth"
        alpha_path = weights_dir / "alpha_controller.pth"

        if unet_path.exists():
            unet.load_state_dict(torch.load(unet_path, map_location="cpu"))
        else:
            raise FileNotFoundError(f"UNet weights not found at {unet_path}")

        if alpha_path.exists():
            alpha_controller.load_state_dict(torch.load(alpha_path, map_location="cpu"))
        else:
            raise FileNotFoundError(f"Alpha controller weights not found at {alpha_path}")

        return cls(
            unet=unet,
            alpha_controller=alpha_controller,
            threshold=threshold,
            device=device,
        )

    def compute_alpha_map(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute alpha map for a batch of images.

        Args:
            images: (B, 3, H, W) - RGB images, normalized

        Returns:
            alpha_maps: (B, 1, H, W) - Alpha maps
        """
        with torch.inference_mode():
            # Reconstruct
            recon = self.unet(images)

            # Compute residual magnitude
            resid_mag = (recon - images).pow(2).mean(dim=1, keepdim=True)

            # Compute redundancy (convert to grayscale first)
            grayscale = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
            redundancy = local_redundancy(grayscale, self.redundancy_window)

            # Predict alpha map
            alpha_maps = self.alpha_controller(resid_mag, redundancy)

        return alpha_maps

    def compute_scores(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores for a batch of images.

        Args:
            images: (B, 3, H, W) - RGB images, normalized

        Returns:
            scores: (B,) - Per-image anomaly scores
        """
        alpha_maps = self.compute_alpha_map(images)

        # Aggregate alpha map to per-image score
        # Higher alpha = more likely corrupted
        scores = torch.sigmoid(alpha_maps).mean(dim=(1, 2, 3))

        return scores

    def check_batch(
        self,
        images: torch.Tensor,
        return_alpha_maps: bool = False,
    ) -> list[CheckResult]:
        """
        Check a batch of images for corruption.

        Args:
            images: (B, 3, H, W) - RGB images, normalized to [0, 1]
            return_alpha_maps: If True, include alpha maps in results

        Returns:
            List of CheckResult objects
        """
        images = images.to(self.device)

        if return_alpha_maps:
            alpha_maps = self.compute_alpha_map(images)
            scores = torch.sigmoid(alpha_maps).mean(dim=(1, 2, 3))

            results = []
            for i, score in enumerate(scores):
                results.append(CheckResult(
                    score=score.item(),
                    flagged=score.item() >= self.threshold,
                    alpha_map=alpha_maps[i].cpu(),
                ))
        else:
            scores = self.compute_scores(images)

            results = []
            for score in scores:
                results.append(CheckResult(
                    score=score.item(),
                    flagged=score.item() >= self.threshold,
                ))

        return results

    def save(self, weights_dir: Union[str, Path]):
        """
        Save model weights to directory.

        Args:
            weights_dir: Directory to save weights
        """
        weights_dir = Path(weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.unet.state_dict(), weights_dir / "unet.pth")
        torch.save(self.alpha_controller.state_dict(), weights_dir / "alpha_controller.pth")
