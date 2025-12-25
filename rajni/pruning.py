"""
Pruning algorithms for RAJNI.

This module contains the core Jacobian-based importance scoring
and adaptive budget computation used for token pruning.

The key insight is that we approximate the Jacobian of the CLS token
w.r.t. patch tokens using attention weights and value norms, avoiding
expensive backpropagation during inference.
"""
import math
from typing import Tuple, Optional
import torch
import torch.nn.functional as F

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
    layer_idx: int = 0
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
    V_mean_heads = values.mean(dim=1)  # [B, N, D]
    V_cls = V_mean_heads[:, 0]  # [B, D]
    
    # Center CLS value (consistent with patch importance centering)
    V_global_mean = V_mean_heads.mean(dim=1)  # [B, D]
    V_cls_centered = V_cls - V_global_mean
    V_cls_norm = V_cls_centered.norm(dim=-1)  # [B]
    
    # Sensitivity: how "anchored" is the CLS token
    rho = (1.0 + A_cls_cls * V_cls_norm)
    rho = calibrate_rho(rho, layer_idx)
    return rho.mean()


# def compute_jacobian_importance(
#     attention: torch.Tensor,
#     values: torch.Tensor,
#     num_patches: int,
#     eps: float = 1e-6,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Compute Jacobian-based importance scores for patch tokens.
    
#     We approximate the Jacobian of CLS w.r.t. patches as:
#         J_i ≈ A[cls→i] * ||V_i - mean(V)||
    
#     This captures both attention flow (which patches CLS attends to)
#     and value saliency (which patches have distinctive features).
    
#     Args:
#         attention: Attention weights [B, H, N, N] after softmax
#         values: Value vectors [B, H, N, D]
#         num_patches: Number of patch tokens (excluding CLS)
#         eps: Small constant for numerical stability
    
#     Returns:
#         importance: Per-patch importance scores [B, num_patches]
#         mass: Total importance mass (scalar, for adaptive budgeting)
#     """
#     # Average attention across heads
#     A_mean = attention.mean(dim=1)  # [B, N, N]
    
#     # CLS-to-patch attention (exclude CLS token at position 0)
#     A_cls_to_patches = A_mean[:, 0, 1:num_patches + 1]  # [B, num_patches]
    
#     # Patch value vectors (averaged across heads)
#     V_patches = values.mean(dim=1)[:, 1:num_patches + 1]  # [B, num_patches, D]
    
#     # Center the value vectors (critical for meaningful norms)
#     V_mean = V_patches.mean(dim=1, keepdim=True)
#     V_centered = V_patches - V_mean
#     V_norm = V_centered.norm(dim=-1)  # [B, num_patches]
    
#     # Standardize within each sample
#     mu = V_norm.mean(dim=1, keepdim=True)
#     std = V_norm.std(dim=1, keepdim=True)
#     V_standardized = (V_norm - mu) / (std + eps)

#     V_cls = values.mean(dim=1)[:, 0]
#     cos_sim = F.cosine_similarity(
#         V_patches, 
#         V_cls.unsqueeze(1), 
#         dim=-1
#     ) 
#     # Jacobian importance: attention * ReLU(standardized value norm)
#     # ReLU ensures we only keep positively salient tokens
#     importance = A_cls_to_patches * F.relu(V_standardized)*(1-cos_sim)
    
#     # Total mass for adaptive budget computation
#     mass = importance.sum(dim=1).mean()
    
#     return importance, mass

def compute_jacobian_importance(
    attention: torch.Tensor,
    values: torch.Tensor,
    num_patches: int,
    eps: float = 1e-6,
    k: int = 5,
    layer_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Jacobian-based importance with LOCAL REDUNDANCY SUPPRESSION.

    Importance_i =
        A(CLS → i)
        × saliency(V_i)
        × (1 − redundancy_kNN(V_i))

    Redundancy is measured using cosine similarity with k nearest
    neighboring patch tokens in value space, after removing
    positional bias via centering.

    Args:
        attention: [B, H, N, N]
        values:    [B, H, N, D]
        num_patches: number of patch tokens (excluding CLS)
        k: number of nearest neighbors for redundancy
        layer_idx: used to disable redundancy in early layers

    Returns:
        importance: [B, num_patches]
        mass: scalar
    """

    # --------------------------------------------------
    # 1. CLS → patch attention
    # --------------------------------------------------
    A_mean = attention.mean(dim=1)                      # [B, N, N]
    A_cls = A_mean[:, 0, 1:num_patches + 1]             # [B, N]

    # --------------------------------------------------
    # 2. Patch value vectors (semantic signal)
    # --------------------------------------------------
    V = values.mean(dim=1)[:, 1:num_patches + 1]        # [B, N, D]

    # ---- remove positional bias (CRITICAL) ----
    V = V - V.mean(dim=1, keepdim=True)

    # --------------------------------------------------
    # 3. Saliency gate (your original logic, untouched)
    # --------------------------------------------------
    V_norm = V.norm(dim=-1)                             # [B, N]

    mu = V_norm.mean(dim=1, keepdim=True)
    std = V_norm.std(dim=1, keepdim=True)

    V_gate = F.relu((V_norm - mu) / (std + eps))        # [B, N]

    # --------------------------------------------------
    # 4. Local redundancy (kNN cosine similarity)
    # --------------------------------------------------
    if layer_idx < 2:
        # Early layers: redundancy signal unreliable
        redundancy = 0.0
    else:
        Vn = F.normalize(V, dim=-1)                     # [B, N, D]

        # cosine similarity matrix
        sim = torch.matmul(Vn, Vn.transpose(-1, -2))    # [B, N, N]

        # top-k neighbors (ignore self at index 0)
        topk_sim, _ = torch.topk(sim, k=k + 1, dim=-1)
        redundancy = topk_sim[:, :, 1:].mean(dim=-1)    # [B, N]
        redundancy = redundancy.clamp(min=0.0, max=1.0)

    # --------------------------------------------------
    # 5. Final importance
    # --------------------------------------------------
    # --------------------------------------------------
    # 5. Layer-adaptive fusion: redundancy → importance
    # --------------------------------------------------
    num_layers = 12  # ViT-Base (pass if you want later)
    alpha = layer_idx / max(num_layers - 1, 1)
    # alpha = alpha.clamp(0.0, 1.0)

    jacobian_score = A_cls * V_gate              # semantic importance
    redundancy_score = (1.0 - redundancy)        # uniqueness

    importance = (
        (alpha * jacobian_score +
        (1.0 - alpha) * redundancy_score)/(jacobian_score+redundancy_score+eps)
    )
    mass = importance.sum(dim=1).mean()

    return importance, mass
def compute_keep_ratio(
    rho: torch.Tensor,
    current_mass: torch.Tensor,
    prev_mass: torch.Tensor,
    gamma: float,
    eps: float = 1e-6,
    layer_idx: int = 1,
    num_layers: int = 12,
    min_factor: float = 0.8,
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
    
    base_keep = torch.exp(-(rho - 0.6) * gamma)
    base_keep = torch.clamp(base_keep, max=1.0)
    # --- Linear layer factor ---
    layer_frac = layer_idx / max(num_layers - 1, 1)
    layer_factor = min_factor + (1.0 - min_factor) * layer_frac

    # --- Final keep ratio ---
    keep_ratio = base_keep * layer_factor
    keep_ratio = torch.clamp(keep_ratio, max=1.0)

    return keep_ratio

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
