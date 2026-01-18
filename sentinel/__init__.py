"""
Sentinel: ML Data Quality Gate

Detects corrupted/degraded images before they hurt model training.
"""

__version__ = "0.1.0"

from .gate import QualityGate, CheckResult
from .models import UNet2D, SpatialAlphaController, local_redundancy
from . import metrics

__all__ = [
    "QualityGate",
    "CheckResult",
    "UNet2D",
    "SpatialAlphaController",
    "local_redundancy",
    "metrics",
]
