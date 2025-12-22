import torch
import numpy as np
import matplotlib.pyplot as plt


@torch.no_grad()
def visualise_pruning(
    model,
    loader,
    img_idx=0,
    patch_size=16,
    device="cuda",
    max_layers=None,
    stats=None,
):
    """
    Visualize RAJNI pruning by explicitly marking PRUNED patches.

    Blue patches = removed tokens.
    This tracks original patch identities across layers.
    """

    model.eval()

    # ---- Load one image ----
    x, _ = next(iter(loader))
    img = x[img_idx].unsqueeze(0).to(device)

    # ---- Forward pass (collect pruning info) ----
    _ = model(img)
    kept_indices = stats["kept_indices"]
    if max_layers is not None:
        kept_indices = kept_indices[:max_layers]

    # ---- Prepare image ----
    img_np = img[0].permute(1, 2, 0).cpu().numpy()
    H, W, _ = img_np.shape

    patches_per_row = W // patch_size
    total_patches = patches_per_row ** 2

    # Track ORIGINAL patch ids
    alive_patch_ids = np.arange(total_patches)

    # ---- Plot ----
    fig, axes = plt.subplots(
        1, len(kept_indices) + 1,
        figsize=(3 * (len(kept_indices) + 1), 3)
    )

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