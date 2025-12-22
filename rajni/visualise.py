import numpy as np
import matplotlib.pyplot as plt


def visualise_pruning(image, kept_indices, patch_size=16):
    """
    Visualizes pruned patches layer-wise.
    """
    H, W, _ = image.shape
    grid = W // patch_size

    fig, axes = plt.subplots(1, len(kept_indices), figsize=(16, 4))

    for i, idx in enumerate(kept_indices):
        mask = np.zeros(grid * grid)
        mask[idx[1:] - 1] = 1  # exclude CLS

        mask = mask.reshape(grid, grid)
        axes[i].imshow(image)
        axes[i].imshow(mask, alpha=0.5)
        axes[i].set_title(f"Layer {i}")
        axes[i].axis("off")

    plt.show()