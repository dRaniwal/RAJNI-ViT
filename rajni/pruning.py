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


def compute_jacobian_importance(
    attention: torch.Tensor,
    values: torch.Tensor,
    num_patches: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Jacobian-based importance with:
    1) semantic centering of value vectors
    2) spatial (positional) debiasing of value norms

    Importance_i = A(CLS → i) × debiased_saliency(V_i)

    Args:
        attention: [B, H, N, N] attention after softmax
        values:    [B, H, N, D] value vectors
        num_patches: number of patch tokens (excluding CLS)

    Returns:
        importance: [B, num_patches]
        mass: scalar importance mass
    """

    # --------------------------------------------------
    # 1. CLS → patch attention
    # --------------------------------------------------
    A_mean = attention.mean(dim=1)                  # [B, N, N]
    A_cls = A_mean[:, 0, 1:num_patches + 1]         # [B, N]

    # --------------------------------------------------
    # 2. Patch value vectors (semantic signal)
    # --------------------------------------------------
    V = values.mean(dim=1)[:, 1:num_patches + 1]    # [B, N, D]

    # ---- semantic centering (removes global bias) ----
    V = V - V.mean(dim=1, keepdim=True)             # [B, N, D]

    # --------------------------------------------------
    # 3. Value norm
    # --------------------------------------------------
    V_norm = V.norm(dim=-1)                         # [B, N]

    # --------------------------------------------------
    # 4. Positional debiasing (CRITICAL FIX)
    # --------------------------------------------------
    B, N = V_norm.shape
    H = W = int(math.sqrt(N))
    assert H * W == N, "Patch count must form square grid"

    V_grid = V_norm.view(B, H, W)                   # [B, H, W]

    # remove absolute spatial bias
    V_grid = V_grid - V_grid.mean(dim=(1, 2), keepdim=True)

    V_debiased = V_grid.view(B, N)                  # [B, N]

    # --------------------------------------------------
    # 5. Saliency gate (centered + debiased)
    # --------------------------------------------------
    mu = V_debiased.mean(dim=1, keepdim=True)
    std = V_debiased.std(dim=1, keepdim=True)

    V_gate = F.relu((V_debiased - mu) / (std + eps))

    # --------------------------------------------------
    # 6. Final Jacobian importance
    # --------------------------------------------------
    importance = A_cls * V_gate                     # [B, N]
    mass = importance.sum(dim=1).mean()

    return importance, mass
# def compute_jacobian_importance(
#     attention: torch.Tensor,
#     values: torch.Tensor,
#     num_patches: int,
#     eps: float = 1e-6,
#     # k: int = 5,
#     # layer_idx: int = 0,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Jacobian-based importance with LOCAL REDUNDANCY SUPPRESSION.

#     Importance_i =
#         A(CLS → i)
#         × saliency(V_i)
#         × (1 − redundancy_kNN(V_i))

#     Redundancy is measured using cosine similarity with k nearest
#     neighboring patch tokens in value space, after removing
#     positional bias via centering.

#     Args:
#         attention: [B, H, N, N]
#         values:    [B, H, N, D]
#         num_patches: number of patch tokens (excluding CLS)
#         k: number of nearest neighbors for redundancy
#         layer_idx: used to disable redundancy in early layers

#     Returns:
#         importance: [B, num_patches]
#         mass: scalar
#     """

#     # --------------------------------------------------
#     # 1. CLS → patch attention
#     # --------------------------------------------------
#     A_mean = attention.mean(dim=1)                      # [B, N, N]
#     A_cls = A_mean[:, 0, 1:num_patches + 1]             # [B, N]

#     # --------------------------------------------------
#     # 2. Patch value vectors (semantic signal)
#     # --------------------------------------------------
#     V = values.mean(dim=1)[:, 1:num_patches + 1]        # [B, N, D]

#     # ---- remove positional bias (CRITICAL) ----
#     V = V - V.mean(dim=1, keepdim=True)

#     # --------------------------------------------------
#     # 3. Saliency gate (your original logic, untouched)
#     # --------------------------------------------------
#     V_norm = V.norm(dim=-1)                             # [B, N]

#     mu = V_norm.mean(dim=1, keepdim=True)
#     std = V_norm.std(dim=1, keepdim=True)

#     V_gate = F.relu((V_norm - mu) / (std + eps))        # [B, N]

#     # --------------------------------------------------
#     # 4. Local redundancy (kNN cosine similarity)
#     # --------------------------------------------------

#     # # Normalize patch vectors
#     # Vn = F.normalize(V, dim=-1)                  # [B, N, D]

#     # # Cosine similarity
#     # sim = torch.matmul(Vn, Vn.transpose(-1, -2)) # [B, N, N]

#     # # k nearest neighbors (exclude self)
#     # topk_sim, topk_idx = torch.topk(sim, k=k + 1, dim=-1)
#     # nbr_sim = topk_sim[:, :, 1:]                 # [B, N, k]
#     # nbr_idx = topk_idx[:, :, 1:]                 # [B, N, k]

#     # # Jacobian base score
#     # jacobian_score = A_cls * V_gate              # [B, N]

#     # # Gather neighbor scores
#     # nbr_score = torch.gather(
#     #     jacobian_score.unsqueeze(-1).expand(-1, -1, k),
#     #     dim=1,
#     #     index=nbr_idx
#     # )                                            # [B, N, k]

#     # # Relative suppression:
#     # # if neighbor is stronger AND similar → suppress
#     # suppression = nbr_sim * (nbr_score / (jacobian_score.unsqueeze(-1) + eps))

#     # redundancy_supp = suppression.max(dim=-1).values
#     # redundancy_supp = redundancy_supp.clamp(0.0, 1.0)

#     # # --------------------------------------------------
#     # # 5. Final importance
#     # # --------------------------------------------------
#     # # --------------------------------------------------
#     # # 5. Layer-adaptive fusion: redundancy → importance
#     # # --------------------------------------------------
#     # num_layers = 12  # ViT-Base (pass if you want later)
#     # alpha = layer_idx / max(num_layers - 1, 1)
#     # # alpha = alpha.clamp(0.0, 1.0)

#     # redundancy_score = (1.0 - redundancy_supp)        # uniqueness

#     jacobian_score = A_cls * V_gate              # semantic importance
#     importance = jacobian_score
#     mass = importance.sum(dim=1).mean()

#     return importance, mass

def compute_keep_ratio(
    rho: torch.Tensor,
    current_mass: torch.Tensor,
    prev_mass: torch.Tensor,
    gamma: float,
    eps: float = 1e-6,
    # layer_idx: int = 1,
    # num_layers: int = 12,
    # min_factor: float = 0.8,
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
    
    # base_keep = torch.exp(-(rho - 0.6)*eta * gamma)
    # base_keep = torch.clamp(base_keep, max=1.0)
    # # --- Linear layer factor ---
    # layer_frac = layer_idx / max(num_layers - 1, 1)
    # layer_factor = min_factor + (1.0 - min_factor) * layer_frac
    # layer_factor = layer_factor**0.5
    base_keep = (rho*eta).clamp(0.25, 4.0)**(-gamma)
    # --- Final keep ratio ---
    keep_ratio = base_keep
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
