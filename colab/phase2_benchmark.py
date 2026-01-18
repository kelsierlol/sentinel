"""
Phase 2: ImageNet-C Benchmark Script for Google Colab

This script benchmarks Sentinel against OpenCV baselines on ImageNet-C corruptions.
Designed to run on Google Colab with GPU.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
from dataclasses import dataclass, asdict

# Import sentinel (will be installed from GitHub in Colab)
from sentinel import QualityGate
from sentinel.models import UNet2D, local_redundancy
from sentinel.metrics import auroc, f1_score, pr_auc
from sentinel.data import corrupt_batch


@dataclass
class BenchmarkResult:
    """Result for a single corruption type."""
    corruption: str
    severity: int
    sentinel_auroc: float
    sentinel_f1: float
    sentinel_pr_auc: float
    opencv_blur_auroc: float
    opencv_blur_f1: float
    opencv_histogram_auroc: float
    opencv_histogram_f1: float
    num_samples: int


class ImageNetCDataset(Dataset):
    """
    ImageNet-C dataset loader.

    Expects structure:
    imagenet-c/
    ├── gaussian_noise/
    │   ├── 1/  # severity 1
    │   ├── 2/
    │   ...
    """

    CORRUPTION_TYPES = [
        "gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        "snow", "frost", "fog", "brightness",
        "contrast", "elastic_transform", "pixelate", "jpeg_compression"
    ]

    def __init__(
        self,
        root: Path,
        corruption_type: str,
        severity: int,
        max_samples: int = 1000,
        transform=None,
    ):
        self.root = Path(root)
        self.corruption_type = corruption_type
        self.severity = severity
        self.transform = transform or self._default_transform()

        self.image_dir = self.root / corruption_type / str(severity)

        # Get all images (ImageNet-C uses .JPEG)
        image_files = list(self.image_dir.glob("*.JPEG"))
        if not image_files:
            # Try PNG as fallback
            image_files = list(self.image_dir.glob("*.png"))

        # Limit to max_samples
        self.image_paths = sorted(image_files)[:max_samples]

        print(f"Loaded {len(self.image_paths)} images for {corruption_type} severity {severity}")

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Label: 1 = corrupted (all ImageNet-C images are corrupted)
        label = 1

        return image, label, str(img_path.name)


class ImageNetCleanDataset(Dataset):
    """Clean ImageNet validation set for FPR testing."""

    def __init__(
        self,
        root: Path,
        max_samples: int = 1000,
        transform=None,
    ):
        self.root = Path(root)
        self.transform = transform or self._default_transform()

        # ImageNet val structure: val/n01440764/ILSVRC2012_val_00000001.JPEG
        self.image_paths = []
        for class_dir in sorted(self.root.glob("n*")):
            if class_dir.is_dir():
                images = list(class_dir.glob("*.JPEG"))
                self.image_paths.extend(images)
                if len(self.image_paths) >= max_samples:
                    break

        self.image_paths = self.image_paths[:max_samples]
        print(f"Loaded {len(self.image_paths)} clean validation images")

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Label: 0 = clean
        label = 0

        return image, label, str(img_path.name)


class OpenCVBaselines:
    """OpenCV-based corruption detectors."""

    @staticmethod
    def blur_score(image: torch.Tensor) -> float:
        """
        Laplacian variance for blur detection.
        Lower variance = more blurry.
        We return 1 - normalized_variance so higher = more corrupted.
        """
        # Convert to numpy
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Normalize: typical clean images have variance ~500-2000
        # Blurry images have variance <100
        normalized = variance / 2000.0
        score = 1.0 - min(normalized, 1.0)  # Invert so high = corrupted

        return score

    @staticmethod
    def histogram_score(image: torch.Tensor) -> float:
        """
        Histogram entropy/uniformity score.
        Very uniform or very peaked histograms indicate corruption.
        """
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()

        # Entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-12))

        # Normal images have entropy ~6-7
        # Corrupted images deviate from this
        target_entropy = 7.0
        score = abs(entropy - target_entropy) / target_entropy

        return score


class AlphaHead2D(nn.Module):
    """Simple alpha head for benchmarking (same as test)."""

    def __init__(self, in_ch: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_sentinel_on_cifar(
    device: torch.device,
    epochs_unet: int = 10,
    epochs_alpha: int = 10,
) -> Tuple[nn.Module, nn.Module]:
    """
    Train Sentinel models on CIFAR-10 for transfer to ImageNet-C.
    Returns (unet, alpha_head).
    """
    from torchvision import datasets
    from torch.utils.data import Subset

    print("\n" + "="*80)
    print("Training Sentinel on CIFAR-10...")
    print("="*80)

    # Load CIFAR-10
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_subset = Subset(train_dataset, range(5000))  # Use 5k samples
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)

    # Initialize models (small UNet for CIFAR-10 32x32)
    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Minimal UNet for 32x32
            self.enc = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
            )
            self.out = nn.Conv2d(64, 3, 1)

        def forward(self, x):
            return torch.sigmoid(self.out(self.enc(x)))

    unet = TinyUNet().to(device)
    alpha_head = AlphaHead2D(in_ch=2, hidden=64).to(device)

    # Phase 1: Train UNet
    print("\nPhase 1: Training UNet...")
    opt_unet = torch.optim.Adam(unet.parameters(), lr=1e-3)
    unet.train()

    for epoch in range(1, epochs_unet + 1):
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            recon = unet(x)
            loss = F.mse_loss(recon, x)

            opt_unet.zero_grad()
            loss.backward()
            opt_unet.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if epoch % 2 == 0:
            print(f"  Epoch {epoch}/{epochs_unet} | Loss: {avg_loss:.6f}")

    # Phase 2: Train Alpha Head
    print("\nPhase 2: Training Alpha Head...")
    opt_alpha = torch.optim.Adam(alpha_head.parameters(), lr=5e-4)
    alpha_head.train()
    unet.eval()

    rng = torch.Generator(device=device).manual_seed(8)

    for epoch in range(1, epochs_alpha + 1):
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)

            # Apply corruptions
            corrupt, mask = corrupt_batch(x, rng,
                                         occlusion_size=8,
                                         blur_size=6,
                                         sp_size=4,
                                         copy_size=6)

            with torch.no_grad():
                recon = unet(corrupt)

            # Compute features
            resid_mag = (recon - x).pow(2).mean(dim=1, keepdim=True).detach()
            grayscale = 0.299 * corrupt[:, 0:1] + 0.587 * corrupt[:, 1:2] + 0.114 * corrupt[:, 2:3]
            redundancy = local_redundancy(grayscale, window=5).detach()

            # Train alpha head
            features = torch.cat([resid_mag, redundancy], dim=1)
            logits = alpha_head(features)

            pos = mask.sum()
            neg = mask.numel() - pos
            pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0)

            loss = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight)

            opt_alpha.zero_grad()
            loss.backward()
            opt_alpha.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if epoch % 2 == 0:
            print(f"  Epoch {epoch}/{epochs_alpha} | Loss: {avg_loss:.4f}")

    print("\nTraining complete!")
    return unet, alpha_head


def evaluate_corruption(
    unet: nn.Module,
    alpha_head: nn.Module,
    dataset: ImageNetCDataset,
    clean_dataset: ImageNetCleanDataset,
    device: torch.device,
    batch_size: int = 32,
) -> BenchmarkResult:
    """
    Evaluate on a single corruption type.
    """
    unet.eval()
    alpha_head.eval()

    # Create combined dataset (corrupted + clean for proper AUROC)
    from torch.utils.data import ConcatDataset
    combined = ConcatDataset([dataset, clean_dataset])
    loader = DataLoader(combined, batch_size=batch_size, shuffle=False)

    sentinel_scores = []
    opencv_blur_scores = []
    opencv_hist_scores = []
    labels = []

    with torch.no_grad():
        for images, batch_labels, _ in loader:
            images = images.to(device)

            # Sentinel scores
            recon = unet(images)
            resid_mag = (recon - images).pow(2).mean(dim=1, keepdim=True)
            grayscale = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
            redundancy = local_redundancy(grayscale, window=5)

            features = torch.cat([resid_mag, redundancy], dim=1)
            logits = alpha_head(features)
            scores = torch.sigmoid(logits).mean(dim=(1, 2, 3))

            sentinel_scores.extend(scores.cpu().numpy())

            # OpenCV baselines
            for img in images:
                opencv_blur_scores.append(OpenCVBaselines.blur_score(img))
                opencv_hist_scores.append(OpenCVBaselines.histogram_score(img))

            labels.extend(batch_labels.numpy())

    # Convert to numpy
    sentinel_scores = np.array(sentinel_scores)
    opencv_blur_scores = np.array(opencv_blur_scores)
    opencv_hist_scores = np.array(opencv_hist_scores)
    labels = np.array(labels)

    # Compute metrics
    result = BenchmarkResult(
        corruption=dataset.corruption_type,
        severity=dataset.severity,
        sentinel_auroc=auroc(sentinel_scores, labels),
        sentinel_f1=f1_score(sentinel_scores, labels, threshold=0.5),
        sentinel_pr_auc=pr_auc(sentinel_scores, labels),
        opencv_blur_auroc=auroc(opencv_blur_scores, labels),
        opencv_blur_f1=f1_score(opencv_blur_scores, labels, threshold=0.5),
        opencv_histogram_auroc=auroc(opencv_hist_scores, labels),
        opencv_histogram_f1=f1_score(opencv_hist_scores, labels, threshold=0.5),
        num_samples=len(labels),
    )

    return result


def run_benchmark(
    imagenet_c_path: str,
    imagenet_val_path: str,
    corruption_types: List[str],
    severity: int = 3,
    max_samples_per_corruption: int = 500,
    max_clean_samples: int = 500,
    output_path: str = "benchmark_results.json",
):
    """
    Run full Phase 2 benchmark.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Train Sentinel on CIFAR-10
    unet, alpha_head = train_sentinel_on_cifar(device, epochs_unet=10, epochs_alpha=10)

    # Load clean dataset once
    print(f"\nLoading clean ImageNet validation set...")
    clean_dataset = ImageNetCleanDataset(
        Path(imagenet_val_path),
        max_samples=max_clean_samples,
    )

    # Run benchmark on each corruption
    results = []

    print(f"\n{'='*80}")
    print(f"Running benchmark on {len(corruption_types)} corruption types at severity {severity}")
    print(f"{'='*80}")

    for corruption in corruption_types:
        print(f"\n--- Evaluating: {corruption} ---")

        try:
            dataset = ImageNetCDataset(
                Path(imagenet_c_path),
                corruption,
                severity,
                max_samples=max_samples_per_corruption,
            )

            result = evaluate_corruption(
                unet, alpha_head, dataset, clean_dataset, device
            )

            results.append(result)

            print(f"  Sentinel  - AUROC: {result.sentinel_auroc:.4f} | F1: {result.sentinel_f1:.4f} | PR-AUC: {result.sentinel_pr_auc:.4f}")
            print(f"  OpenCV Blur - AUROC: {result.opencv_blur_auroc:.4f} | F1: {result.opencv_blur_f1:.4f}")
            print(f"  OpenCV Hist - AUROC: {result.opencv_histogram_auroc:.4f} | F1: {result.opencv_histogram_f1:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save results
    results_dict = {
        "corruption_results": [asdict(r) for r in results],
        "summary": {
            "num_corruptions": len(results),
            "severity": severity,
            "sentinel_mean_auroc": np.mean([r.sentinel_auroc for r in results]),
            "opencv_blur_mean_auroc": np.mean([r.opencv_blur_auroc for r in results]),
            "opencv_hist_mean_auroc": np.mean([r.opencv_histogram_auroc for r in results]),
        }
    }

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Sentinel mean AUROC: {results_dict['summary']['sentinel_mean_auroc']:.4f}")
    print(f"  OpenCV Blur mean AUROC: {results_dict['summary']['opencv_blur_mean_auroc']:.4f}")
    print(f"  OpenCV Hist mean AUROC: {results_dict['summary']['opencv_hist_mean_auroc']:.4f}")
    print(f"\n  Improvement over OpenCV Blur: {(results_dict['summary']['sentinel_mean_auroc'] - results_dict['summary']['opencv_blur_mean_auroc']) * 100:.1f}%")

    return results_dict


if __name__ == "__main__":
    # Example usage (will be called from Colab notebook)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-c", type=str, required=True)
    parser.add_argument("--imagenet-val", type=str, required=True)
    parser.add_argument("--corruption-types", type=str, default="gaussian_noise,defocus_blur,motion_blur")
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--output", type=str, default="benchmark_results.json")

    args = parser.parse_args()

    corruption_types = [c.strip() for c in args.corruption_types.split(",")]

    run_benchmark(
        args.imagenet_c,
        args.imagenet_val,
        corruption_types,
        severity=args.severity,
        output_path=args.output,
    )
