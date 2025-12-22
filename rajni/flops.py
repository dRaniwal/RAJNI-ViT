def baseline_vit_flops(model, tokens):
    """
    Approximate baseline ViT FLOPs per forward.
    """
    L = len(model.blocks)
    D = model.embed_dim
    return L * (tokens ** 2) * D


def adaptive_vit_flops(token_counts, embed_dim):
    """
    FLOPs used by RAJNI based on token counts.
    """
    flops = 0
    for t in token_counts:
        flops += (t ** 2) * embed_dim
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