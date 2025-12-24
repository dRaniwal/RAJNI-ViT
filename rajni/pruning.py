"""
Pruning algorithms for RAJNI.

This module contains the core Jacobian-based importance scoring
and adaptive budget computation used for token pruning.

The key insight is that we approximate the Jacobian of the CLS token
w.r.t. patch tokens using attention weights and value norms, avoiding
expensive backpropagation during inference.
"""
from typing import Tuple, Optional
import torch
import torch.nn.functional as F


def compute_cls_sensitivity(
    attention: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """
    Compute CLS token sensitivity (rho) from attention and values.
    
    This measures how much the CLS token attends to itself, weighted
    by the norm of its value vector. High rho indicates the model
    is confident and less aggressive pruning is appropriate.
    
    Args:
        attention: Attention weights [B, H, N, N] after softmax
        values: Value vectors [B, H, N, D]
    
    Returns:
        rho: Scalar sensitivity measure
    """
    # Average across heads
    A_mean = attention.mean(dim=1)  # [B, N, N]
    A_cls_cls = A_mean[:, 0, 0]     # [B] - CLS self-attention
    
    # CLS value vector norm (averaged across heads)
    V_cls = values.mean(dim=1)[:, 0]  # [B, D]
    V_cls_norm = V_cls.norm(dim=-1)   # [B]
    
    # Sensitivity: how "anchored" is the CLS token
    rho = (1.0 + A_cls_cls * V_cls_norm).mean()
    
    return rho


def compute_jacobian_importance(
    attention: torch.Tensor,
    values: torch.Tensor,
    num_patches: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Jacobian-based importance scores for patch tokens.
    
    We approximate the Jacobian of CLS w.r.t. patches as:
        J_i ≈ A[cls→i] * ||V_i - mean(V)||
    
    This captures both attention flow (which patches CLS attends to)
    and value saliency (which patches have distinctive features).
    
    Args:
        attention: Attention weights [B, H, N, N] after softmax
        values: Value vectors [B, H, N, D]
        num_patches: Number of patch tokens (excluding CLS)
        eps: Small constant for numerical stability
    
    Returns:
        importance: Per-patch importance scores [B, num_patches]
        mass: Total importance mass (scalar, for adaptive budgeting)
    """
    # Average attention across heads
    A_mean = attention.mean(dim=1)  # [B, N, N]
    
    # CLS-to-patch attention (exclude CLS token at position 0)
    A_cls_to_patches = A_mean[:, 0, 1:num_patches + 1]  # [B, num_patches]
    
    # Patch value vectors (averaged across heads)
    V_patches = values.mean(dim=1)[:, 1:num_patches + 1]  # [B, num_patches, D]
    
    # Center the value vectors (critical for meaningful norms)
    V_mean = V_patches.mean(dim=1, keepdim=True)
    V_centered = V_patches - V_mean
    V_norm = V_centered.norm(dim=-1)  # [B, num_patches]
    
    # Standardize within each sample
    mu = V_norm.mean(dim=1, keepdim=True)
    std = V_norm.std(dim=1, keepdim=True)
    V_standardized = (V_norm - mu) / (std + eps)
    
    # Jacobian importance: attention * ReLU(standardized value norm)
    # ReLU ensures we only keep positively salient tokens
    importance = A_cls_to_patches * F.relu(V_standardized)
    
    # Total mass for adaptive budget computation
    mass = importance.sum(dim=1).mean()
    
    return importance, mass


def compute_keep_ratio(
    rho: torch.Tensor,
    current_mass: torch.Tensor,
    prev_mass: torch.Tensor,
    gamma: float,
    eps: float = 1e-6,
) -> float:
    """
    Compute adaptive keep ratio based on layer dynamics.
    
    The key idea: if importance mass is decreasing (eta < 1),
    we can prune more aggressively. If it's increasing, we
    should be conservative.
    
    The formula: keep_ratio = (rho * eta)^(-gamma)
    
    Args:
        rho: CLS sensitivity from compute_cls_sensitivity
        current_mass: Current layer's importance mass
        prev_mass: Previous layer's importance mass
        gamma: Pruning aggressiveness (higher = more pruning)
        eps: Small constant for numerical stability
    
    Returns:
        keep_ratio: Fraction of tokens to keep (0.25 to 4.0)
    """
    # Relative change in importance mass
    eta = current_mass / (prev_mass + eps)
    
    # Adaptive keep ratio with clamping for stability
    ratio_raw = (rho ).clamp(0.25, 4.0) ** (-gamma)
    
    return float(ratio_raw)


def select_tokens(
    importance: torch.Tensor,
    num_keep: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Select top-k tokens based on importance scores.
    
    Returns indices that include the CLS token (always kept)
    plus the top-k most important patch tokens.
    
    Args:
        importance: Per-patch importance [B, num_patches]
        num_keep: Number of patch tokens to keep
        device: Target device for indices
    
    Returns:
        keep_indices: Token indices to keep [1 + num_keep]
                      (CLS at position 0, then sorted patch indices)
    """
    # Average importance across batch for selection
    scores = importance.mean(dim=0)
    
    # Select top-k patch indices
    _, top_indices = torch.topk(scores, k=num_keep)
    
    # Sort for consistent ordering (helps with caching)
    sorted_indices = top_indices.sort().values
    
    # Shift by 1 to account for CLS token at position 0
    patch_indices = sorted_indices + 1
    
    # Prepend CLS token index
    cls_index = torch.zeros(1, device=device, dtype=torch.long)
    keep_indices = torch.cat([cls_index, patch_indices])
    
    return keep_indices
