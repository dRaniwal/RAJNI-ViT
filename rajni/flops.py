def baseline_vit_flops(model, num_tokens):
    """
    Approximate baseline ViT FLOPs (attention + MLP).
    """
    L = len(model.blocks)
    D = model.embed_dim
    H = model.blocks[0].attn.num_heads
    N = num_tokens

    # Attention: QK + AV
    attn_flops = 2 * L * H * N * N * (D // H)

    # MLP: 2 linear layers
    mlp_flops = 2 * L * N * D * (4 * D)

    return attn_flops + mlp_flops


def adaptive_flops(token_counts, embed_dim, num_heads):
    """
    Compute FLOPs actually used based on remaining tokens per layer.
    """
    flops = 0
    D = embed_dim
    H = num_heads

    for N in token_counts:
        attn = 2 * H * N * N * (D // H)
        mlp = 2 * N * D * (4 * D)
        flops += attn + mlp

    return flops


def flops_reduction(model):
    """
    Compare baseline FLOPs vs RAJNI FLOPs.
    """
    base_tokens = model.last_token_counts[0]
    baseline = baseline_vit_flops(model.m, base_tokens)
    used = adaptive_flops(
        model.last_token_counts,
        model.embed_dim,
        model.num_heads,
    )

    return {
        "baseline_flops": baseline,
        "rajni_flops": used,
        "reduction_pct": 100.0 * (1.0 - used / baseline),
    }