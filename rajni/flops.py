def baseline_vit_flops(model, tokens):
    """
    Proper ViT FLOPs: Attention + FFN
    """
    L = len(model.blocks)
    D = model.embed_dim

    flops_per_layer = (
        12 * tokens * D * D +   # QKV + proj + FFN
        2 * tokens * tokens * D
    )

    return L * flops_per_layer


def adaptive_vit_flops(token_counts, embed_dim):
    """
    FLOPs used by RAJNI based on per-layer token counts.
    """
    D = embed_dim
    flops = 0

    for N in token_counts:
        flops += (
            12 * N * D * D +
            2 * N * N * D
        )

    return flops



def flops_reduction(model, stats):
    """
    Compare baseline vs RAJNI FLOPs.
    """
    token_counts = stats["token_counts"]
    base_tokens = token_counts[0]

    baseline = baseline_vit_flops(model, base_tokens)
    used = adaptive_vit_flops(token_counts, model.embed_dim)

    return {
        "baseline_GFLOPs": baseline / 1e9,
        "rajni_GFLOPs": used / 1e9,
        "reduction_%": 100 * (1 - used / baseline),
    }