import numpy as np


def print_token_counts(model):
    """
    Print layer-wise token counts from the last forward pass.
    """
    print("\nLayer-wise token counts:")
    for i, n in enumerate(model.last_token_counts):
        print(f"Layer {i:02d}: {n} tokens")


def unwrap_model(model):
    """
    Safely unwrap DataParallel / DistributedDataParallel models.

    This is standard practice in research codebases (ToMe, MAE, timm).
    """
    return model.module if hasattr(model, "module") else model


def denormalize_image(img_np, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Denormalize an image from ImageNet normalization.

    Args:
        img_np: numpy array [H, W, 3] normalized with ImageNet stats
        mean: ImageNet mean (default: ImageNet values)
        std: ImageNet std (default: ImageNet values)

    Returns:
        Denormalized image clipped to [0, 1]
    """
    mean = np.array(mean)
    std = np.array(std)
    img_np = img_np * std + mean
    return np.clip(img_np, 0, 1)