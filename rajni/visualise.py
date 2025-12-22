import matplotlib.pyplot as plt
import numpy as np


def visualize_pruning(image, kept_indices, patch_size=16):
    """
    Simple visualization of pruned patches.
    Blue = pruned, Original = kept.
    """
    img = image.permute(1, 2, 0).cpu().numpy()
    H, W, _ = img.shape

    grid = W // patch_size
    mask = np.ones((grid, grid))

    if kept_indices is not None:
        kept = kept_indices[1:] - 1  # remove CLS
        for idx in range(grid * grid):
            if idx not in kept:
                r = idx // grid
                c = idx % grid
                mask[r, c] = 0

    plt.imshow(img)
    for r in range(grid):
        for c in range(grid):
            if mask[r, c] == 0:
                plt.gca().add_patch(
                    plt.Rectangle(
                        (c * patch_size, r * patch_size),
                        patch_size,
                        patch_size,
                        fill=True,
                        color="blue",
                        alpha=0.4,
                    )
                )
    plt.axis("off")
    plt.show()