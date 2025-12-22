from .utils import unwrap_model


def baseline_vit_flops(model, tokens):
    """
    Rough ViT FLOPs estimate for comparison.
    """
    D = model.embed_dim
    L = len(model.blocks)

    attn = L * (tokens ** 2) * D
    mlp = L * tokens * (D ** 2)

    return attn + mlp


def adaptive_flops(token_counts, embed_dim):
    """
    FLOPs based on actual tokens used per layer.
    """
    flops = 0
    for t in token_counts:
        flops += (t ** 2) * embed_dim
        flops += t * (embed_dim ** 2)
    return flops


def flops_reduction(model):
    model = unwrap_model(model)

    assert hasattr(model, "last_token_counts"), \
        "Run a forward pass before computing FLOPs."

    base_tokens = model.last_token_counts[0]
    D = model.m.embed_dim

    baseline = baseline_vit_flops(model.m, base_tokens)
    used = adaptive_flops(model.last_token_counts, D)

    return {
        "baseline_GFLOPs": baseline / 1e9,
        "rajni_GFLOPs": used / 1e9,
        "reduction_%": 100 * (1 - used / baseline),
    }