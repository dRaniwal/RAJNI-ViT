"""
Utility functions for RAJNI.

Common helpers for model handling and image processing.
"""
from typing import Tuple, Union
import numpy as np
import torch.nn as nn


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Unwrap a model from DataParallel or DistributedDataParallel.
    
    This is standard practice in research codebases (ToMe, MAE, timm)
    to access the underlying model for statistics and analysis.
    
    Args:
        model: Potentially wrapped PyTorch model
    
    Returns:
        The underlying model without parallelization wrapper
    """
    if hasattr(model, "module"):
        return model.module
    return model


def denormalize_image(
    img: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Denormalize an image for visualization.
    
    Reverses the normalization typically applied during preprocessing,
    converting from normalized tensor values back to displayable range.
    
    Args:
        img: Normalized image array [H, W, 3]
        mean: Per-channel mean used in normalization
        std: Per-channel std used in normalization
    
    Returns:
        Denormalized image clipped to [0, 1]
    """
    mean_arr = np.array(mean)
    std_arr = np.array(std)
    img = img * std_arr + mean_arr
    return np.clip(img, 0.0, 1.0)


def get_patch_grid_size(image_size: int, patch_size: int) -> int:
    """
    Calculate number of patches per side for a given image/patch size.
    
    Args:
        image_size: Input image dimension (assumes square)
        patch_size: Size of each patch
    
    Returns:
        Number of patches per side
    """
    return image_size // patch_size
