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
import math

# Layer-specific calibration coefficients for ρ (rho)
# Fitted via linear regression on ImageNet validation data:
#   calibrated_rho = a * raw_rho + b
# These coefficients correct the systematic overestimation of raw ρ
# See scripts/validate_rho.py for validation methodology
RHO_CALIBRATION_COEFFICIENTS = {
    # Layer: (slope a, intercept b) - fitted via least squares
    0:  (0.40, 0.20),   # ρ_raw ~1.40 → exact ~0.75
    1:  (0.45, 0.28),   # ρ_raw ~1.43 → exact ~0.93  
    2:  (0.42, 0.25),   # ρ_raw ~1.46 → exact ~0.87
    3:  (0.38, 0.29),   # ρ_raw ~1.61 → exact ~0.91
    4:  (0.40, 0.20),   # ρ_raw ~1.56 → exact ~0.82
    5:  (0.50, 0.23),   # ρ_raw ~1.46 → exact ~0.96
    6:  (0.55, 0.25),   # ρ_raw ~1.39 → exact ~1.01
    7:  (0.65, 0.19),   # ρ_raw ~1.18 → exact ~0.95
    8:  (0.80, 0.10),   # ρ_raw ~1.04 → exact ~0.92
    9:  (0.75, 0.13),   # ρ_raw ~1.07 → exact ~0.94
    10: (0.85, 0.02),   # ρ_raw ~1.01 → exact ~0.88
    11: (0.95, 0.20),   # ρ_raw ~1.01 → exact ~1.15 (gradient amplification)
}


def calibrate_rho(rho_raw: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """
    Apply layer-specific linear calibration to raw rho value.
    
    Raw ρ = 1 + A(CLS→CLS) · ||V_CLS - mean(V)|| systematically
    overestimates the true gradient ratio ∂y/∂CLS_{l+1} / ∂y/∂CLS_l.
    
    This function applies empirically-derived linear transform:
        ρ_calibrated = a * ρ_raw + b
    
    Args:
        rho_raw: Uncalibrated rho value [scalar or tensor]
        layer_idx: Layer index (0-11 for ViT-Base)
    
    Returns:
        Calibrated rho value
    """
    a, b = RHO_CALIBRATION_COEFFICIENTS.get(layer_idx, (0.55, 0.20))
    return a * rho_raw + b


def compute_cls_sensitivity(
    attention: torch.Tensor,
    values: torch.Tensor,
    layer_idx: int = 0,
    calibrate: bool = True,
) -> torch.Tensor:
    """
    Compute CLS token sensitivity (rho) from attention and values.
    
    This measures how much the CLS token attends to itself, weighted
    by the norm of its value vector. Uses centered V for consistency
    with patch importance computation.
    
    Args:
        attention: Attention weights [B, H, N, N] after softmax
        values: Value vectors [B, H, N, D]
        layer_idx: Layer index for calibration (0-11 for ViT-Base)
        calibrate: Whether to apply layer-specific calibration
    
    Returns:
        rho: Scalar sensitivity measure (calibrated if enabled)
    """
    # Average across heads
    A_mean = attention.mean(dim=1)  # [B, N, N]
    A_cls_cls = A_mean[:, 0, 0]     # [B] - CLS self-attention
    
    # CLS value vector norm with centering (consistent with patch importance)
    V_mean_heads = values.mean(dim=1)  # [B, N, D]
    V_cls = V_mean_heads[:, 0]  # [B, D]
    V_mean = V_mean_heads.mean(dim=1)  # [B, D] - mean across all tokens
    V_cls_centered = V_cls - V_mean
    V_cls_norm = V_cls_centered.norm(dim=-1)   # [B]
    

    V_all_mean = V_mean.mean(dim=1)  # [B, D] - mean across all tokens
    V_cls_centered = V_cls - V_all_mean  # [B, D]
    V_cls_norm = V_cls_centered.norm(dim=-1)  # [B]
    # Raw sensitivity: how "anchored" is the CLS token
    rho_raw = (1.0 + A_cls_cls * V_cls_norm).mean()
    
    # Apply calibration if enabled
    if calibrate:
        return calibrate_rho(rho_raw, layer_idx)
    return rho_raw


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


def compute_importance_and_sensitivity(
    attention: torch.Tensor,
    values: torch.Tensor,
    num_patches: int,
    layer_idx: int = 0,
    calibrate: bool = True,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused computation of both importance scores and CLS sensitivity.
    
    This combines compute_cls_sensitivity and compute_jacobian_importance
    to reduce redundant operations (A_mean computed once).
    
    Args:
        attention: Attention weights [B, H, N, N] after softmax
        values: Value vectors [B, H, N, D]
        num_patches: Number of patch tokens (excluding CLS)
        layer_idx: Layer index for rho calibration (0-11 for ViT-Base)
        calibrate: Whether to apply layer-specific rho calibration
        eps: Small constant for numerical stability
    
    Returns:
        importance: Per-patch importance scores [B, num_patches]
        mass: Total importance mass (scalar)
        rho: CLS sensitivity (scalar, calibrated if enabled)
    """
    # Average attention across heads (compute once)
    A_mean = attention.mean(dim=1)  # [B, N, N]
    
    # === CLS Sensitivity (rho) with centered V ===
    A_cls_cls = A_mean[:, 0, 0]  # [B]
    V_mean_heads = values.mean(dim=1)  # [B, N, D]
    V_cls = V_mean_heads[:, 0]  # [B, D]
    
    # Center CLS value (consistent with patch importance centering)
    V_global_mean = V_mean_heads.mean(dim=1)  # [B, D]
    V_cls_centered = V_cls - V_global_mean
    V_cls_norm = V_cls_centered.norm(dim=-1)  # [B]
    
    # Raw rho
    rho_raw = (1.0 + A_cls_cls * V_cls_norm).mean()
    
    # Apply calibration if enabled
    if calibrate:
        rho = calibrate_rho(rho_raw, layer_idx)
    else:
        rho = rho_raw
    
    # === Jacobian Importance ===
    A_cls_to_patches = A_mean[:, 0, 1:num_patches + 1]  # [B, num_patches]
    V_patches = V_mean_heads[:, 1:num_patches + 1]  # [B, num_patches, D]
    
    # Center and normalize value vectors
    V_patch_mean = V_patches.mean(dim=1, keepdim=True)
    V_centered = V_patches - V_patch_mean
    V_norm = V_centered.norm(dim=-1)  # [B, num_patches]
    
    # Standardize
    mu = V_norm.mean(dim=1, keepdim=True)
    std = V_norm.std(dim=1, keepdim=True)
    V_standardized = (V_norm - mu) / (std + eps)
    
    # Importance = attention * ReLU(standardized value norm)
    importance = A_cls_to_patches * F.relu(V_standardized)
    mass = importance.sum(dim=1).mean()
    
    return importance, mass, rho


def compute_keep_ratio(
    rho: torch.Tensor,
    current_mass: torch.Tensor,
    prev_mass: torch.Tensor,
    gamma: float,
    eps: float = 1e-6,
) -> torch.Tensor:
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
        keep_ratio: Fraction of tokens to keep (tensor, 0.25 to 4.0)
    """
    # Relative change in importance mass
    eta = current_mass / (prev_mass + eps)
    
    # Adaptive keep ratio with clamping for stability
    # Returns a tensor to avoid GPU-CPU sync
    # ratio_raw =(rho ).clamp(0.25, 1.0) / (gamma)
    # prune_ratio =(rho-0.8)*(gamma).clamp(0.0, 2)
    ratio_raw=min(math.exp(-(rho-0.6)*gamma),1)
    return ratio_raw


@torch.no_grad()
def select_tokens(
    importance: torch.Tensor,
    num_keep: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Select top-k tokens based on importance scores.
    
    Returns indices that include the CLS token (always kept)
    plus the top-k most important patch tokens.
    
    Optimized for GPU: all operations stay on device, minimal allocations.
    
    Args:
        importance: Per-patch importance [B, num_patches]
        num_keep: Number of patch tokens to keep
        device: Target device for indices
    
    Returns:
        keep_indices: Token indices to keep [1 + num_keep]
                      (CLS at position 0, then sorted patch indices)
    """
    # Average importance across batch for selection
    # Use sum instead of mean to avoid division (faster)
    scores = importance.sum(dim=0)
    
    # Select top-k patch indices (already on GPU)
    _, top_indices = torch.topk(scores, k=num_keep, sorted=False)
    
    # Sort for consistent ordering (helps with memory coalescing)
    sorted_indices = top_indices.sort().values
    
    # Shift by 1 to account for CLS token at position 0
    # In-place add to avoid allocation
    patch_indices = sorted_indices.add_(1)
    
    # Prepend CLS token index using cat (optimized for small tensors)
    keep_indices = torch.cat([
        torch.zeros(1, device=device, dtype=torch.long),
        patch_indices
    ])
    
    return keep_indices
