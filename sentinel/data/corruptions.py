"""Synthetic corruption utilities for training and testing."""

import torch
import torch.nn.functional as F
from typing import Tuple


def apply_blur_patch(x: torch.Tensor, top: int, left: int, size: int) -> torch.Tensor:
    kernel = torch.tensor(
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=x.dtype, device=x.device
    )
    kernel = kernel / kernel.sum()
    k = kernel.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
    blurred = F.conv2d(x, k, padding=1, groups=x.size(1))
    x[:, :, top:top + size, left:left + size].copy_(blurred[:, :, top:top + size, left:left + size])
    return x


def corrupt_batch(
    x: torch.Tensor,
    rng: torch.Generator,
    occlusion_size: int = 4,
    blur_size: int = 3,
    sp_size: int = 2,
    copy_size: int = 3,
    modes: Tuple[str, ...] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply synthetic corruptions to a batch of images.

    Args:
        x: (B, C, H, W) - Input images
        rng: Random number generator for reproducibility
        occlusion_size: Size of occlusion patches
        blur_size: Size of blur patches
        sp_size: Size of salt & pepper patches
        copy_size: Size of copy-paste patches
        modes: Corruption types to apply

    Returns:
        corrupted: (B, C, H, W) - Corrupted images
        mask: (B, 1, H, W) - Binary mask (1 = corrupted region)
    """
    bsz, _, h, w = x.shape
    mask = torch.zeros((bsz, 1, h, w), device=x.device)
    corrupted = x.clone()

    def rand(shape, device):
        if rng is not None and device.type == "cpu":
            return torch.rand(shape, generator=rng, device=device)
        return torch.rand(shape, device=device)

    def randint(low, high):
        device = x.device
        if rng is not None:
            return torch.randint(low, high, (1,), generator=rng, device=device).item()
        return torch.randint(low, high, (1,), device=device).item()

    if modes is None:
        modes = ("occlusion", "blur", "saltpepper", "copy")

    for i in range(bsz):
        # Occlusion square
        if "occlusion" in modes:
            top = randint(0, h - occlusion_size)
            left = randint(0, w - occlusion_size)
            corrupted[i, :, top:top + occlusion_size, left:left + occlusion_size] = 0.0
            mask[i, 0, top:top + occlusion_size, left:left + occlusion_size] = 1.0

        # Blur patch
        if "blur" in modes:
            top = randint(0, h - blur_size)
            left = randint(0, w - blur_size)
            corrupted[i:i + 1] = apply_blur_patch(corrupted[i:i + 1], top, left, blur_size)
            mask[i, 0, top:top + blur_size, left:left + blur_size] = 1.0

        # Salt & pepper patch
        if "saltpepper" in modes:
            top = randint(0, h - sp_size)
            left = randint(0, w - sp_size)
            sp = rand((sp_size, sp_size), device=x.device)
            sp = (sp > 0.5).float()
            corrupted[i, :, top:top + sp_size, left:left + sp_size] = sp.unsqueeze(0).repeat(corrupted.size(1), 1, 1)
            mask[i, 0, top:top + sp_size, left:left + sp_size] = 1.0

        # Copy-paste patch
        if "copy" in modes:
            src_top = randint(0, h - copy_size)
            src_left = randint(0, w - copy_size)
            dst_top = randint(0, h - copy_size)
            dst_left = randint(0, w - copy_size)
            corrupted[i, :, dst_top:dst_top + copy_size, dst_left:dst_left + copy_size] = \
                corrupted[i, :, src_top:src_top + copy_size, src_left:src_left + copy_size]
            mask[i, 0, dst_top:dst_top + copy_size, dst_left:dst_left + copy_size] = 1.0

    return corrupted, mask
