"""
Visualization utilities for RAJNI pruning.

Provides functions to visualize which patches are pruned at each layer.
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def extract_normalize_stats(loader: DataLoader) -> Tuple[Optional[tuple], Optional[tuple]]:
    """
    Extract normalization statistics from a DataLoader's transform.
    
    Searches the transform pipeline for a Normalize transform and
    extracts its mean and std values.
    
    Args:
        loader: DataLoader with transforms applied
    
    Returns:
        mean: Normalization mean as tuple, or None
        std: Normalization std as tuple, or None
    """
    dataset = loader.dataset
    
    # Unwrap dataset wrappers (Subset, etc.)
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    
    transform = getattr(dataset, 'transform', None)
    if transform is None:
        return None, None
    
    # Handle Compose transforms
    if hasattr(transform, 'transforms'):
        transforms_list = transform.transforms
    else:
        transforms_list = [transform]
    
    # Find Normalize transform
    for t in transforms_list:
        if t.__class__.__name__ == 'Normalize':
            mean = getattr(t, 'mean', None)
            std = getattr(t, 'std', None)
            if mean is not None and std is not None:
                # Convert to tuple
                if hasattr(mean, 'tolist'):
                    mean = tuple(mean.tolist())
                elif isinstance(mean, list):
                    mean = tuple(mean)
                if hasattr(std, 'tolist'):
                    std = tuple(std.tolist())
                elif isinstance(std, list):
                    std = tuple(std)
                return mean, std
    
    return None, None


def denormalize_image(
    img: np.ndarray,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> np.ndarray:
    """Denormalize an image for visualization."""
    mean_arr = np.array(mean)
    std_arr = np.array(std)
    img = img * std_arr + mean_arr
    return np.clip(img, 0.0, 1.0)


@torch.no_grad()
def visualize_pruning(
    model: nn.Module,
    image: Optional[torch.Tensor] = None,
    loader: Optional[DataLoader] = None,
    img_idx: int = 0,
    patch_size: int = 16,
    device: str = "cuda",
    max_layers: Optional[int] = None,
    stats: Optional[Dict[str, Any]] = None,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize RAJNI pruning by highlighting removed patches.
    
    Creates a multi-panel figure showing the original image and
    the pruning state at each layer. Pruned patches are shown in blue.
    
    Args:
        model: RAJNI-wrapped model
        image: Pre-loaded image tensor [1, 3, H, W] or [3, H, W]
        loader: DataLoader to get image from (if image not provided)
        img_idx: Index of image in batch (if using loader)
        patch_size: ViT patch size (default: 16)
        device: Target device
        max_layers: Maximum layers to visualize (None = all)
        stats: Pre-computed stats (if None, runs forward pass)
        mean: Normalization mean (auto-extracted if None)
        std: Normalization std (auto-extracted if None)
        save_path: If provided, save figure to this path
    """
    model.eval()
    
    # Get image
    if image is not None:
        if image.dim() == 3:
            img_tensor = image.unsqueeze(0).to(device)
        else:
            img_tensor = image.to(device)
    elif loader is not None:
        batch, _ = next(iter(loader))
        img_tensor = batch[img_idx].unsqueeze(0).to(device)
    else:
        raise ValueError("Must provide either 'image' or 'loader'")
    
    # Auto-extract normalization stats
    if (mean is None or std is None) and loader is not None:
        extracted_mean, extracted_std = extract_normalize_stats(loader)
        mean = mean or extracted_mean
        std = std or extracted_std
    
    # Run forward pass if stats not provided
    if stats is None:
        _ = model(img_tensor)
        if hasattr(model, "module"):
            stats = model.module.get_last_stats()
        else:
            stats = model.get_last_stats()
    
    kept_indices = stats["kept_indices"]
    if max_layers is not None:
        kept_indices = kept_indices[:max_layers]
    
    # Prepare image for display
    img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    if mean is not None and std is not None:
        img_np = denormalize_image(img_np, mean, std)
    
    H, W, _ = img_np.shape
    patches_per_row = W // patch_size
    total_patches = patches_per_row ** 2
    
    # Track surviving patches through layers
    alive_patches = np.arange(total_patches)
    
    # Create figure
    n_panels = len(kept_indices) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3))
    
    if n_panels == 1:
        axes = [axes]
    
    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    # Each pruning layer
    for layer_idx, keep_idx in enumerate(kept_indices):
        ax = axes[layer_idx + 1]
        ax.imshow(img_np)
        ax.set_title(f"Layer {layer_idx + 1}")
        ax.axis("off")
        
        # Map kept indices to original patch positions
        keep_rel = keep_idx[keep_idx != 0].cpu().numpy() - 1
        valid_mask = (keep_rel >= 0) & (keep_rel < len(alive_patches))
        keep_rel = keep_rel[valid_mask]
        
        alive_patches = alive_patches[keep_rel]
        pruned = set(range(total_patches)) - set(alive_patches.tolist())
        
        # Draw pruned patches
        for p in pruned:
            row = p // patches_per_row
            col = p % patches_per_row
            rect = plt.Rectangle(
                (col * patch_size, row * patch_size),
                patch_size,
                patch_size,
                linewidth=0,
                facecolor="blue",
                alpha=0.35,
            )
            ax.add_patch(rect)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
