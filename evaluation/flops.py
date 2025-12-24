"""
FLOPs computation for Vision Transformers.

Provides theoretical FLOP counts for comparing pruned vs unpruned models.
The computation focuses on the dominant operations: attention and FFN.
"""
from typing import Dict, List, Any
import torch.nn as nn


def compute_baseline_flops(model: nn.Module, num_tokens: int) -> int:
    """
    Compute theoretical FLOPs for a standard ViT forward pass.
    
    Accounts for the main compute-heavy operations:
    - QKV projection: 3 * N * D * D
    - Attention: 2 * N * N * D (QK^T and AV)
    - Projection: N * D * D
    - FFN: 8 * N * D * D (two linear layers with 4x expansion)
    
    Total per layer: 12 * N * D^2 + 2 * N^2 * D
    
    Args:
        model: Vision Transformer model (must have .blocks and .embed_dim)
        num_tokens: Number of tokens (including CLS)
    
    Returns:
        Total FLOPs for one forward pass
    """
    L = len(model.blocks)
    D = model.embed_dim
    N = num_tokens
    
    # Per-layer FLOPs
    flops_linear = 12 * N * D * D   # QKV + proj + FFN
    flops_attention = 2 * N * N * D  # Attention matrix ops
    
    flops_per_layer = flops_linear + flops_attention
    
    return L * flops_per_layer


def compute_adaptive_flops(token_counts: List[int], embed_dim: int) -> int:
    """
    Compute FLOPs for a RAJNI forward pass with varying token counts.
    
    Since RAJNI prunes tokens progressively, later layers operate on
    fewer tokens, reducing overall compute.
    
    Args:
        token_counts: Number of tokens at each layer
        embed_dim: Model embedding dimension
    
    Returns:
        Total FLOPs for one forward pass
    """
    D = embed_dim
    total_flops = 0
    
    for N in token_counts:
        flops_linear = 12 * N * D * D
        flops_attention = 2 * N * N * D
        total_flops += flops_linear + flops_attention
    
    return total_flops


def flops_reduction(
    model: nn.Module,
    stats: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compare baseline vs RAJNI FLOPs and compute reduction.
    
    This is the main function for reporting FLOPs savings in papers
    and experiments.
    
    Args:
        model: The base ViT model (not wrapped)
        stats: Pruning statistics from model.get_last_stats()
    
    Returns:
        Dictionary with:
        - baseline_GFLOPs: Baseline model FLOPs in billions
        - rajni_GFLOPs: RAJNI model FLOPs in billions
        - reduction_percent: Percentage reduction
    
    Example:
        >>> stats = pruned_model.get_last_stats()
        >>> core = unwrap_model(pruned_model).m
        >>> result = flops_reduction(core, stats)
        >>> print(f"FLOPs reduction: {result['reduction_percent']:.1f}%")
    """
    token_counts = stats["token_counts"]
    initial_tokens = token_counts[0]
    
    baseline = compute_baseline_flops(model, initial_tokens)
    adaptive = compute_adaptive_flops(token_counts, model.embed_dim)
    
    reduction = 100.0 * (1.0 - adaptive / baseline)
    
    return {
        "baseline_GFLOPs": baseline / 1e9,
        "rajni_GFLOPs": adaptive / 1e9,
        "reduction_percent": reduction,
    }
