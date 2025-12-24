"""
RAJNI: Relative Adaptive Jacobian-based Neuronal Importance.

Efficient Vision Transformers via adaptive token pruning at inference time.

This package provides:
- AdaptiveJacobianPrunedViT: Main model wrapper for token pruning
- Pruning utilities for Jacobian-based importance scoring
- Helper functions for model handling and visualization

Example:
    >>> import timm
    >>> from rajni import AdaptiveJacobianPrunedViT
    >>> 
    >>> base = timm.create_model('vit_base_patch16_224', pretrained=True)
    >>> model = AdaptiveJacobianPrunedViT(base, gamma=0.01)
    >>> logits = model(images)
"""

__version__ = "0.1.0"
__author__ = "Dhairya Raniwal"

from .model import AdaptiveJacobianPrunedViT
from .utils import unwrap_model, denormalize_image

__all__ = [
    "AdaptiveJacobianPrunedViT",
    "unwrap_model",
    "denormalize_image",
]