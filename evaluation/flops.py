"""
FLOPs computation for Vision Transformers.

Provides theoretical FLOP counts for comparing pruned vs unpruned models.
The computation focuses on the dominant operations: attention and FFN.

Note: We count multiply-accumulate operations (MACs) as 2 FLOPs each,
which is the standard convention in papers like ToMe, DynamicViT, etc.
"""
from typing import Dict, List, Any
import torch.nn as nn


def compute_patch_embed_flops(model: nn.Module, image_size: int = 224) -> int:
    """
    Compute FLOPs for patch embedding layer.
    
    For a conv2d with kernel=patch_size, stride=patch_size:
    FLOPs = 2 * C_in * C_out * H_out * W_out * K * K
    
    For ViT-B with patch_size=16, image=224:
    = 2 * 3 * 768 * 14 * 14 * 16 * 16 = 1.16B
    """
    patch_embed = model.patch_embed
    
    # Get dimensions
    patch_size = patch_embed.patch_size[0] if hasattr(patch_embed.patch_size, '__getitem__') else patch_embed.patch_size
    embed_dim = model.embed_dim
    in_channels = 3
    
    # Output spatial dimensions
    H_out = W_out = image_size // patch_size
    
    # Conv2d FLOPs: 2 * C_in * C_out * H_out * W_out * K_h * K_w
    flops = 2 * in_channels * embed_dim * H_out * W_out * patch_size * patch_size
    
    return flops


def compute_head_flops(model: nn.Module) -> int:
    """
    Compute FLOPs for classification head.
    
    Simple linear layer: 2 * D * num_classes
    """
    embed_dim = model.embed_dim
    num_classes = model.head.out_features if hasattr(model.head, 'out_features') else 1000
    
    return 2 * embed_dim * num_classes


def compute_layer_flops(num_tokens: int, embed_dim: int, mlp_ratio: float = 4.0) -> int:
    """
    Compute FLOPs for a single transformer layer.
    
    Breakdown:
    - QKV projection: 2 * N * D * 3D = 6 * N * D^2
    - Attention scores (Q @ K^T): 2 * N * N * D
    - Attention output (A @ V): 2 * N * N * D  
    - Output projection: 2 * N * D * D
    - FFN layer 1: 2 * N * D * (mlp_ratio * D)
    - FFN layer 2: 2 * N * (mlp_ratio * D) * D
    
    Total: (8 + 2*mlp_ratio*2) * N * D^2 + 4 * N^2 * D
         = (8 + 16) * N * D^2 + 4 * N^2 * D  (for mlp_ratio=4)
         = 24 * N * D^2 + 4 * N^2 * D
    
    Note: Previous formula (12*N*D^2 + 2*N^2*D) only counted MACs, not FLOPs.
    Here we count FLOPs (1 MAC = 2 FLOPs).
    """
    N = num_tokens
    D = embed_dim
    
    # Attention
    qkv_proj = 6 * N * D * D       # 3 projections
    attn_scores = 2 * N * N * D    # Q @ K^T
    attn_values = 2 * N * N * D    # A @ V
    out_proj = 2 * N * D * D       # output projection
    
    # FFN
    ffn_ratio = int(mlp_ratio)
    ffn_up = 2 * N * D * D * ffn_ratio      # D -> 4D
    ffn_down = 2 * N * D * ffn_ratio * D    # 4D -> D
    
    total = qkv_proj + attn_scores + attn_values + out_proj + ffn_up + ffn_down
    
    return total


def compute_baseline_flops(model: nn.Module, num_tokens: int) -> int:
    """
    Compute theoretical FLOPs for a standard ViT forward pass.
    
    Args:
        model: Vision Transformer model (must have .blocks and .embed_dim)
        num_tokens: Number of tokens (including CLS)
    
    Returns:
        Total FLOPs for one forward pass
    """
    L = len(model.blocks)
    D = model.embed_dim
    
    # Get MLP ratio from first block
    mlp = model.blocks[0].mlp
    if hasattr(mlp, 'fc1'):
        mlp_ratio = mlp.fc1.out_features / D
    else:
        mlp_ratio = 4.0
    
    # Patch embedding
    patch_flops = compute_patch_embed_flops(model)
    
    # Transformer layers
    layer_flops = compute_layer_flops(num_tokens, D, mlp_ratio)
    transformer_flops = L * layer_flops
    
    # Classification head
    head_flops = compute_head_flops(model)
    
    return patch_flops + transformer_flops + head_flops


def compute_adaptive_flops(
    token_counts: List[int], 
    embed_dim: int,
    mlp_ratio: float = 4.0,
    patch_embed_flops: int = 0,
    head_flops: int = 0,
) -> int:
    """
    Compute FLOPs for a RAJNI forward pass with varying token counts.
    
    Since RAJNI prunes tokens progressively, later layers operate on
    fewer tokens, reducing overall compute.
    
    Args:
        token_counts: Number of tokens at each layer
        embed_dim: Model embedding dimension
        mlp_ratio: FFN expansion ratio (default 4.0)
        patch_embed_flops: FLOPs for patch embedding (fixed cost)
        head_flops: FLOPs for classification head (fixed cost)
    
    Returns:
        Total FLOPs for one forward pass
    """
    total_flops = patch_embed_flops + head_flops
    
    for N in token_counts:
        total_flops += compute_layer_flops(N, embed_dim, mlp_ratio)
    
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
    
    # Get MLP ratio
    D = model.embed_dim
    mlp = model.blocks[0].mlp
    if hasattr(mlp, 'fc1'):
        mlp_ratio = mlp.fc1.out_features / D
    else:
        mlp_ratio = 4.0
    
    # Fixed costs (same for both)
    patch_flops = compute_patch_embed_flops(model)
    head_flops = compute_head_flops(model)
    
    baseline = compute_baseline_flops(model, initial_tokens)
    adaptive = compute_adaptive_flops(
        token_counts, D, mlp_ratio, patch_flops, head_flops
    )
    
    reduction = 100.0 * (1.0 - adaptive / baseline)
    
    return {
        "baseline_GFLOPs": baseline / 1e9,
        "rajni_GFLOPs": adaptive / 1e9,
        "reduction_percent": reduction,
    }
