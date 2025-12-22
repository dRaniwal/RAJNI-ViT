def print_token_counts(model):
    """
    Print layer-wise token counts from the last forward pass.
    """
    print("\nLayer-wise token counts:")
    for i, n in enumerate(model.last_token_counts):
        print(f"Layer {i:02d}: {n} tokens")