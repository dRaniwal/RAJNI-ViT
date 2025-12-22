import torch
import numpy as np
import matplotlib.pyplot as plt

from .utils import denormalize_image


def extract_normalize_stats(loader):
    """
    Extract mean and std from a DataLoader's transform pipeline.
    
    Searches for torchvision.transforms.Normalize in the transform chain.
    Returns None, None if not found.
    """
    dataset = loader.dataset
    
    # Handle common dataset wrappers
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    
    transform = getattr(dataset, 'transform', None)
    if transform is None:
        return None, None
    
    # Handle Compose
    transforms_list = []
    if hasattr(transform, 'transforms'):
        transforms_list = transform.transforms
    else:
        transforms_list = [transform]
    
    # Search for Normalize transform
    for t in transforms_list:
        # Check class name to avoid import dependency
        if t.__class__.__name__ == 'Normalize':
            mean = getattr(t, 'mean', None)
            std = getattr(t, 'std', None)
            if mean is not None and std is not None:
                # Convert to tuple if tensor/list
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


@torch.no_grad()
def visualise_pruning(
    model,
    loader,
    img_idx=0,
    patch_size=16,
    device="cuda",
    max_layers=None,
    stats=None,
    denormalize=True,
    mean=None,
    std=None,
):
    """
    Visualize RAJNI pruning by explicitly marking PRUNED patches.

    Blue patches = removed tokens.
    This tracks original patch identities across layers.

    Args:
        model: The RAJNI model
        loader: DataLoader to get images from
        img_idx: Index of image in batch to visualize
        patch_size: Size of ViT patches (default 16)
        device: Device to run on
        max_layers: Limit visualization to first N layers
        stats: Pre-computed stats (if None, will get from model)
        denormalize: Whether to denormalize images
        mean: Normalization mean (auto-extracted from loader if None)
        std: Normalization std (auto-extracted from loader if None)
    """

    model.eval()

    # ---- Auto-extract normalization stats if not provided ----
    if denormalize and (mean is None or std is None):
        extracted_mean, extracted_std = extract_normalize_stats(loader)
        if mean is None:
            mean = extracted_mean
        if std is None:
            std = extracted_std
        
        # If still None, skip denormalization
        if mean is None or std is None:
            print("Warning: Could not extract normalization stats from loader. "
                  "Skipping denormalization. Pass mean/std manually if needed.")
            denormalize = False

    # ---- Load one image ----
    x, _ = next(iter(loader))
    img = x[img_idx].unsqueeze(0).to(device)

    # ---- Forward pass (collect pruning info) ----
    _ = model(img)

    # Get stats from model if not provided
    if stats is None:
        if hasattr(model, "module"):
            stats = model.module.get_last_stats()
        else:
            stats = model.get_last_stats()

    kept_indices = stats["kept_indices"]
    if max_layers is not None:
        kept_indices = kept_indices[:max_layers]

    # ---- Prepare image ----
    img_np = img[0].permute(1, 2, 0).cpu().numpy()

    # Denormalize for proper display
    if denormalize:
        img_np = denormalize_image(img_np, mean, std)

    H, W, _ = img_np.shape

    patches_per_row = W // patch_size
    total_patches = patches_per_row ** 2

    # Track ORIGINAL patch ids
    alive_patch_ids = np.arange(total_patches)

    # ---- Plot ----
    n_plots = len(kept_indices) + 1
    fig, axes = plt.subplots(
        1, n_plots,
        figsize=(3 * n_plots, 3)
    )

    # Handle single subplot case
    if n_plots == 1:
        axes = [axes]

    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for l, keep_idx in enumerate(kept_indices):
        ax = axes[l + 1]
        ax.imshow(img_np)
        ax.set_title(f"Layer {l+1}")
        ax.axis("off")

        # Remove CLS and map to original patch ids
        keep_rel = keep_idx[keep_idx != 0].cpu().numpy() - 1

        # Filter valid indices
        valid_mask = (keep_rel >= 0) & (keep_rel < len(alive_patch_ids))
        keep_rel = keep_rel[valid_mask]

        alive_patch_ids = alive_patch_ids[keep_rel]

        pruned = set(range(total_patches)) - set(alive_patch_ids.tolist())

        for p in pruned:
            r = p // patches_per_row
            c = p % patches_per_row

            rect = plt.Rectangle(
                (c * patch_size, r * patch_size),
                patch_size,
                patch_size,
                linewidth=0,
                facecolor="blue",
                alpha=0.35,
            )
            ax.add_patch(rect)

    plt.tight_layout()
    plt.show()