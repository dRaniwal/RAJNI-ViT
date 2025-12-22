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