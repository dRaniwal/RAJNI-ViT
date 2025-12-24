"""
Pruning algorithms for RAJNI.

This module implements Jacobian-based token importance estimation
for Vision Transformers using attention and value statistics.

Core idea:
We approximate the Jacobian of the CLS token w.r.t. patch tokens as

    J_i ≈ A(CLS → i) · || V_i − mean(V) ||

This avoids backpropagation at inference time while preserving
faithful saliency structure.
"""

from typing import Tuple
import torch
import torch.nn.functional as F
import math

# ---------------------------------------------------------------------
# Layer-wise rho calibration (empirically fitted)
# ---------------------------------------------------------------------

RHO_CALIBRATION_COEFFICIENTS = {
    0:  (0.40, 0.20),
    1:  (0.45, 0.28),
    2:  (0.42, 0.25),
    3:  (0.38, 0.29),
    4:  (0.40, 0.20),
    5:  (0.50, 0.23),
    6:  (0.55, 0.25),
    7:  (0.65, 0.19),
    8:  (0.80, 0.10),
    9:  (0.75, 0.13),
    10: (0.85, 0.02),
    11: (0.95, 0.20),
}


def calibrate_rho(rho_raw: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """
    Linear calibration of CLS sensitivity rho.
    """
    a, b = RHO_CALIBRATION_COEFFICIENTS.get(layer_idx, (0.55, 0.20))
    return a * rho_raw + b


# ---------------------------------------------------------------------
# CLS sensitivity (rho)
# ---------------------------------------------------------------------

def compute_cls_sensitivity(
    attention: torch.Tensor,
    values: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """
    Compute CLS sensitivity:

        rho = 1 + A(CLS → CLS) · || V_CLS − mean(V) ||

    This estimates how strongly the CLS token propagates forward.
    """

    # Head-averaged attention
    A_mean = attention.mean(dim=1)         # [B, N, N]
    A_cls_cls = A_mean[:, 0, 0]             # [B]

    # Head-averaged values
    V_mean_heads = values.mean(dim=1)       # [B, N, D]
    V_global_mean = V_mean_heads.mean(dim=1)  # [B, D]

    V_cls = V_mean_heads[:, 0]              # [B, D]
    V_cls_centered = V_cls - V_global_mean
    V_cls_norm = V_cls_centered.norm(dim=-1)

    rho_raw = 1.0 + A_cls_cls * V_cls_norm
    rho = calibrate_rho(rho_raw, layer_idx)

    return rho


# ---------------------------------------------------------------------
# Jacobian-based patch importance (DERIVATION-FAITHFUL)
# ---------------------------------------------------------------------

def compute_jacobian_importance(
    attention: torch.Tensor,
    values: torch.Tensor,
    num_patches: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-patch Jacobian importance.

    Importance:
        I_i = A(CLS → i) · || V_i − mean(V_patches) ||

    No clipping, no standardization — preserves spatial saliency.
    """

    # Average attention across heads
    A_mean = attention.mean(dim=1)  # [B, N, N]

    # CLS → patch attention
    A_cls_to_patches = A_mean[:, 0, 1:num_patches + 1]  # [B, P]

    # Patch value vectors (head-averaged)
    V_patches = values.mean(dim=1)[:, 1:num_patches + 1]  # [B, P, D]

    # Center value vectors (patch-wise)
    V_patch_mean = V_patches.mean(dim=1, keepdim=True)    # [B, 1, D]
    V_centered = V_patches - V_patch_mean

    # Value magnitude
    V_norm = V_centered.norm(dim=-1)  # [B, P]

    # Jacobian importance
    importance = A_cls_to_patches * V_norm  # [B, P]

    # Importance mass (used only for adaptive budgeting)
    mass = importance.sum(dim=1).mean()

    return importance, mass


# ---------------------------------------------------------------------
# Fused importance + sensitivity (single pass)
# ---------------------------------------------------------------------

def compute_importance_and_sensitivity(
    attention: torch.Tensor,
    values: torch.Tensor,
    num_patches: int,
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Joint computation to avoid redundant ops.
    """

    A_mean = attention.mean(dim=1)  # [B, N, N]

    # ---- CLS sensitivity ----
    A_cls_cls = A_mean[:, 0, 0]

    V_mean_heads = values.mean(dim=1)        # [B, N, D]
    V_global_mean = V_mean_heads.mean(dim=1) # [B, D]

    V_cls = V_mean_heads[:, 0]
    V_cls_centered = V_cls - V_global_mean
    V_cls_norm = V_cls_centered.norm(dim=-1)

    rho_raw = 1.0 + A_cls_cls * V_cls_norm
    rho = calibrate_rho(rho_raw, layer_idx).mean()

    # ---- Patch importance ----
    A_cls_to_patches = A_mean[:, 0, 1:num_patches + 1]
    V_patches = V_mean_heads[:, 1:num_patches + 1]

    V_patch_mean = V_patches.mean(dim=1, keepdim=True)
    V_centered = V_patches - V_patch_mean
    V_norm = V_centered.norm(dim=-1)

    importance = A_cls_to_patches * V_norm
    mass = importance.sum(dim=1).mean()

    return importance, mass, rho


# ---------------------------------------------------------------------
# Adaptive pruning ratio
# ---------------------------------------------------------------------

def compute_keep_ratio(
    rho: torch.Tensor,
    gamma: float,
) -> float:
    """
    Conservative, monotonic pruning schedule.

    Higher rho ⇒ keep more tokens
    Higher gamma ⇒ prune more aggressively
    """
    return min(math.exp(-(rho - 0.6) * gamma), 1.0)


# ---------------------------------------------------------------------
# Token selection
# ---------------------------------------------------------------------

@torch.no_grad()
def select_tokens(
    importance: torch.Tensor,
    num_keep: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Select top-k patch tokens + CLS.
    """

    # Batch-aggregated importance (stable & fast)
    scores = importance.sum(dim=0)  # [P]

    _, top_idx = torch.topk(scores, k=num_keep, sorted=False)
    top_idx = top_idx.sort().values + 1  # shift for CLS

    keep_indices = torch.cat([
        torch.zeros(1, device=device, dtype=torch.long),
        top_idx
    ])

    return keep_indices