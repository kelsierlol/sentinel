"""Test Sentinel on CIFAR-10 to validate the fixed architecture."""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

from sentinel.models import UNet2D, local_redundancy
from sentinel.data import corrupt_batch
from sentinel.metrics import pr_auc


class AlphaHead2D(nn.Module):
    """Alpha head for training (unclamped logits for BCE)."""

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


def set_seed(seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TinyUNet2D(nn.Module):
    """Smaller UNet for CIFAR-10 (32x32)."""

    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()
        # Simple 2-level encoder-decoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.GroupNorm(4, base),
            nn.SiLU(),
        )
        self.down1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)

        self.mid = nn.Sequential(
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.SiLU(),
        )

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )

        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.GroupNorm(4, base),
            nn.SiLU(),
        )

        self.out = nn.Conv2d(base, in_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        m = self.mid(d2)

        u2 = self.up2(m)
        c2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(c2)
        c1 = self.dec1(torch.cat([u1, e1], dim=1))

        return torch.sigmoid(self.out(c1))


def main():
    print("=" * 80)
    print("SENTINEL CIFAR-10 VALIDATION TEST")
    print("Testing AlphaHead2D with 3x3 convs (proven architecture)")
    print("Target: PR-AUC >= 0.70")
    print("=" * 80)

    set_seed(7)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\nDevice: {device}")

    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("\nLoading CIFAR-10...")
    cifar_path = "/Users/prajwal/Projects/supervised_rl/alpha_map_vision/data"
    train_dataset = datasets.CIFAR10(root=cifar_path, train=True, download=False, transform=transform)
    train_subset = Subset(train_dataset, range(2000))  # Use 2000 samples for quick test
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)

    test_dataset = datasets.CIFAR10(root=cifar_path, train=False, download=False, transform=transform)
    test_subset = Subset(test_dataset, range(500))  # 500 test samples
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    print(f"Train samples: {len(train_subset)}")
    print(f"Test samples: {len(test_subset)}")

    # Initialize models
    print("\nInitializing models...")
    unet = TinyUNet2D(in_ch=3, base=32).to(device)
    alpha_head = AlphaHead2D(in_ch=2, hidden=64).to(device)

    print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")
    print(f"Alpha head parameters: {sum(p.numel() for p in alpha_head.parameters()):,}")

    # Phase 1: Train UNet
    print("\n" + "=" * 80)
    print("PHASE 1: Training UNet for reconstruction")
    print("=" * 80)

    opt_unet = torch.optim.Adam(unet.parameters(), lr=1e-3)
    unet.train()

    epochs_unet = 5
    for epoch in range(1, epochs_unet + 1):
        total_loss = 0.0
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            recon = unet(x)
            loss = F.mse_loss(recon, x)

            opt_unet.zero_grad()
            loss.backward()
            opt_unet.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs_unet} | Loss: {avg_loss:.6f}")

    # Phase 2: Train Alpha Controller
    print("\n" + "=" * 80)
    print("PHASE 2: Training Alpha Controller on synthetic corruptions")
    print("=" * 80)

    opt_alpha = torch.optim.Adam(alpha_head.parameters(), lr=5e-4)
    alpha_head.train()
    unet.eval()

    rng = torch.Generator(device=device).manual_seed(8)

    epochs_alpha = 5
    for epoch in range(1, epochs_alpha + 1):
        total_loss = 0.0
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)

            # Apply synthetic corruptions (larger patches for 32x32 images)
            corrupt, mask = corrupt_batch(x, rng,
                                         occlusion_size=8,
                                         blur_size=6,
                                         sp_size=4,
                                         copy_size=6)

            with torch.no_grad():
                recon = unet(corrupt)

            # Compute residual magnitude against CLEAN target (not corrupt input)
            # This measures reconstruction error - high where corruption exists
            resid_mag = (recon - x).pow(2).mean(dim=1, keepdim=True).detach()

            # Compute redundancy
            grayscale = 0.299 * corrupt[:, 0:1] + 0.587 * corrupt[:, 1:2] + 0.114 * corrupt[:, 2:3]
            redundancy = local_redundancy(grayscale, window=5).detach()

            # Concatenate features
            features = torch.cat([resid_mag, redundancy], dim=1)

            # Predict logits (unclamped)
            logits = alpha_head(features)

            # Compute pos_weight for class imbalance
            pos = mask.sum()
            neg = mask.numel() - pos
            pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0)

            # Train to predict corruption mask (BCE with logits handles sigmoid internally)
            loss = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight)

            opt_alpha.zero_grad()
            loss.backward()
            opt_alpha.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs_alpha} | Loss: {avg_loss:.4f}")

    # Phase 3: Evaluation
    print("\n" + "=" * 80)
    print("PHASE 3: Evaluation on test set")
    print("=" * 80)

    unet.eval()
    alpha_head.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)

            # Apply corruptions (same sizes as training)
            corrupt, mask = corrupt_batch(x, rng,
                                         occlusion_size=8,
                                         blur_size=6,
                                         sp_size=4,
                                         copy_size=6)

            # Reconstruct
            recon = unet(corrupt)

            # Compute features (residual vs CLEAN target)
            resid_mag = (recon - x).pow(2).mean(dim=1, keepdim=True)
            grayscale = 0.299 * corrupt[:, 0:1] + 0.587 * corrupt[:, 1:2] + 0.114 * corrupt[:, 2:3]
            redundancy = local_redundancy(grayscale, window=5)

            # Predict logits
            features = torch.cat([resid_mag, redundancy], dim=1)
            logits = alpha_head(features)
            score = torch.sigmoid(logits)

            all_scores.append(score.cpu().numpy())
            all_labels.append(mask.cpu().numpy())

    # Compute metrics
    all_scores = np.concatenate(all_scores, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    pr_auc_score = pr_auc(all_scores, all_labels)

    print(f"\n{'=' * 80}")
    print(f"RESULTS")
    print(f"{'=' * 80}")
    print(f"Pixel-level PR-AUC: {pr_auc_score:.4f}")
    print(f"\nTarget: >= 0.70 (from kpis.md)")

    if pr_auc_score >= 0.70:
        print(f"\n✅ SUCCESS! Architecture fix validated.")
        print(f"   PR-AUC {pr_auc_score:.4f} >= 0.70 target")
        print(f"\n   The 3x3 spatial convs are working!")
    elif pr_auc_score >= 0.50:
        print(f"\n⚠️  PARTIAL SUCCESS: {pr_auc_score:.4f}")
        print(f"   Better than random (0.50) but below target (0.70)")
        print(f"   May need more epochs or hyperparameter tuning")
    else:
        print(f"\n❌ FAILED: {pr_auc_score:.4f} < 0.50")
        print(f"   Still not learning spatial patterns correctly")

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
