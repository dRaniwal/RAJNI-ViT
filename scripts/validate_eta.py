#!/usr/bin/env python3
"""
RAJNI η (eta) and Full Pruning Schedule Validation Script

This script validates RAJNI's approximation of both:
1. ρ (rho) - CLS sensitivity: 1 + A(CLS→CLS) · ||V_CLS||
2. η (eta) - Information mass ratio: M_l / M_{l-1}
3. Combined: keep_ratio = (ρ · η)^(-γ)

It computes exact Jacobian values via backpropagation and compares to approximations.

Theory:
- ρ_l · η_l gives us the ratio of average token sensitivity across adjacent layers
- Pruning happens only when both are stable (product > 1)

Usage:
    python scripts/validate_eta.py
    python scripts/validate_eta.py --gamma 0.05 --num_samples 10
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Dict, Tuple, Optional


def forward_with_jacobian_capture(
    model: nn.Module,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Forward pass capturing CLS tokens and patch tokens at each layer.
    
    Returns:
        logits: Model output
        cls_tokens: CLS token at each layer (before block)
        patch_tokens: Patch tokens at each layer (before block)
        attn_values: (attention, values) at each layer
    """
    x = model.patch_embed(x)
    x = model._pos_embed(x)
    x = model.patch_drop(x)
    
    cls_tokens = []
    patch_tokens = []
    attn_values = []
    
    num_heads = model.blocks[0].attn.num_heads
    head_dim = model.embed_dim // num_heads
    scale = model.blocks[0].attn.scale
    
    for i, blk in enumerate(model.blocks):
        # Store tokens BEFORE this block
        cls_token = x[:, 0].clone()
        cls_token.retain_grad()
        cls_tokens.append(cls_token)
        
        patches = x[:, 1:].clone()
        patches.retain_grad()
        patch_tokens.append(patches)
        
        # Manual attention
        B, N, C = x.shape
        x_norm = blk.norm1(x)
        
        qkv = blk.attn.qkv(x_norm)
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        
        attn_values.append((attn.detach().clone(), v.detach().clone()))
        
        attn_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        attn_out = blk.attn.proj(attn_out)
        attn_out = blk.attn.proj_drop(attn_out)
        
        x = x + blk.drop_path1(attn_out)
        x = x + blk.drop_path2(blk.mlp(blk.norm2(x)))
    
    # Final tokens
    cls_token = x[:, 0].clone()
    cls_token.retain_grad()
    cls_tokens.append(cls_token)
    
    patches = x[:, 1:].clone()
    patches.retain_grad()
    patch_tokens.append(patches)
    
    # Classification
    x = model.norm(x)
    logits = model.head(x[:, 0])
    
    return logits, cls_tokens, patch_tokens, attn_values


def compute_exact_patch_jacobians(
    model: nn.Module,
    x: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Compute exact ∂y/∂CLS_l and ∂y/∂patch_l via backpropagation.
    
    Returns:
        cls_grads: Gradient w.r.t. CLS at each layer
        patch_grads: Gradient w.r.t. patches at each layer
        attn_values: Attention and values at each layer
    """
    model.eval()
    x = x.clone().requires_grad_(True)
    
    logits, cls_tokens, patch_tokens, attn_values = forward_with_jacobian_capture(model, x)
    
    # Use predicted class
    target_class = logits.argmax(dim=-1)
    target = torch.zeros_like(logits)
    target.scatter_(1, target_class.unsqueeze(1), 1.0)
    
    y = (logits * target).sum()
    y.backward(retain_graph=True)
    
    cls_grads = [t.grad.clone() if t.grad is not None else None for t in cls_tokens]
    patch_grads = [t.grad.clone() if t.grad is not None else None for t in patch_tokens]
    
    return cls_grads, patch_grads, attn_values


def compute_rajni_metrics(
    attn: torch.Tensor,
    values: torch.Tensor,
    num_patches: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute RAJNI's approximations:
    - rho: 1 + A(CLS→CLS) · ||V_CLS||
    - importance: A(CLS→patch) · g(||V_patch||)
    - mass: sum of importance
    """
    # Average across heads
    A_mean = attn.mean(dim=1)  # [B, N, N]
    
    # === rho ===
    A_cls_cls = A_mean[:, 0, 0]  # [B]
    V_mean = values.mean(dim=1)  # [B, N, D]
    V_cls = V_mean[:, 0]  # [B, D]
    V_cls_norm = V_cls.norm(dim=-1)  # [B]
    rho = 1.0 + A_cls_cls * V_cls_norm
    
    # === importance ===
    A_cls_to_patches = A_mean[:, 0, 1:num_patches + 1]  # [B, num_patches]
    V_patches = V_mean[:, 1:num_patches + 1]  # [B, num_patches, D]
    
    # Center values
    V_patch_mean = V_patches.mean(dim=1, keepdim=True)
    V_centered = V_patches - V_patch_mean
    V_norm = V_centered.norm(dim=-1)  # [B, num_patches]
    
    # Standardize
    mu = V_norm.mean(dim=1, keepdim=True)
    std = V_norm.std(dim=1, keepdim=True)
    V_std = (V_norm - mu) / (std + eps)
    
    # Importance
    importance = A_cls_to_patches * F.relu(V_std)
    mass = importance.sum(dim=1)  # [B]
    
    return rho, importance, mass


def validate_full_schedule(
    model_name: str = "vit_base_patch16_224",
    gamma: float = 0.05,
    batch_size: int = 2,
    num_samples: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
):
    """
    Validate ρ, η, and the full pruning schedule.
    """
    torch.manual_seed(seed)
    
    print("=" * 80)
    print("RAJNI Full Pruning Schedule Validation (ρ, η, keep_ratio)")
    print("=" * 80)
    print(f"\nModel: {model_name}")
    print(f"Gamma: {gamma}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Num samples: {num_samples}")
    print()
    
    # Load model
    print("Loading model...")
    model = timm.create_model(model_name, pretrained=True)
    model = model.to(device)
    model.eval()
    
    num_layers = len(model.blocks)
    num_patches = (224 // 16) ** 2  # 196 for ViT-B/16
    print(f"Number of layers: {num_layers}")
    print(f"Number of patches: {num_patches}")
    print()
    
    # Storage for all samples
    all_results = {
        'exact_cls_ratio': [[] for _ in range(num_layers)],
        'rajni_rho': [[] for _ in range(num_layers)],
        'exact_patch_mass': [[] for _ in range(num_layers + 1)],
        'exact_eta': [[] for _ in range(num_layers)],
        'rajni_mass': [[] for _ in range(num_layers)],
        'rajni_eta': [[] for _ in range(num_layers)],
        'exact_product': [[] for _ in range(num_layers)],
        'rajni_product': [[] for _ in range(num_layers)],
        'keep_ratio': [[] for _ in range(num_layers)],
    }
    
    for sample_idx in range(num_samples):
        print(f"\n{'─' * 80}")
        print(f"Sample {sample_idx + 1}/{num_samples}")
        print(f"{'─' * 80}")
        
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        
        # Compute exact gradients
        cls_grads, patch_grads, attn_values = compute_exact_patch_jacobians(model, x)
        
        # === Compute exact metrics ===
        
        # CLS gradient norms
        cls_grad_norms = []
        for grad in cls_grads:
            if grad is not None:
                cls_grad_norms.append(grad.norm(dim=-1).mean().item())
            else:
                cls_grad_norms.append(0.0)
        
        # Patch gradient "mass" (sum of gradient norms)
        patch_masses = []
        for grad in patch_grads:
            if grad is not None:
                # ||∂y/∂patch_t|| for each patch, then sum
                patch_norms = grad.norm(dim=-1)  # [B, num_patches]
                mass = patch_norms.sum(dim=1).mean().item()  # Mean across batch
                patch_masses.append(mass)
            else:
                patch_masses.append(0.0)
        
        # === Compute RAJNI metrics ===
        rajni_rhos = []
        rajni_masses = []
        
        for layer in range(num_layers):
            attn, v = attn_values[layer]
            rho, importance, mass = compute_rajni_metrics(attn, v, num_patches)
            rajni_rhos.append(rho.mean().item())
            rajni_masses.append(mass.mean().item())
        
        # === Print per-layer comparison ===
        print(f"\n  {'Layer':^6}│{'Exact CLS':^12}│{'Exact':^10}│{'RAJNI':^10}│{'Exact':^12}│{'RAJNI':^12}│{'Exact':^12}│{'RAJNI':^12}│{'Keep':^10}")
        print(f"  {'':^6}│{'Grad Norm':^12}│{'Ratio':^10}│{'ρ':^10}│{'Patch Mass':^12}│{'Mass':^12}│{'η':^12}│{'η':^12}│{'Ratio':^10}")
        print(f"  {'─' * 6}┼{'─' * 12}┼{'─' * 10}┼{'─' * 10}┼{'─' * 12}┼{'─' * 12}┼{'─' * 12}┼{'─' * 12}┼{'─' * 10}")
        
        prev_rajni_mass = None
        prev_patch_mass = None
        
        for layer in range(num_layers):
            # Exact CLS ratio
            if cls_grad_norms[layer] > 1e-10:
                exact_cls_ratio = cls_grad_norms[layer + 1] / cls_grad_norms[layer]
            else:
                exact_cls_ratio = float('nan')
            
            # Exact eta (patch mass ratio)
            if prev_patch_mass is not None and prev_patch_mass > 1e-10:
                exact_eta = patch_masses[layer + 1] / prev_patch_mass
            else:
                exact_eta = 1.0
            
            # RAJNI eta
            if prev_rajni_mass is not None and prev_rajni_mass > 1e-10:
                rajni_eta = rajni_masses[layer] / prev_rajni_mass
            else:
                rajni_eta = 1.0
            
            # Products
            if not (exact_cls_ratio != exact_cls_ratio):  # Check NaN
                exact_product = exact_cls_ratio * exact_eta
            else:
                exact_product = float('nan')
            
            rajni_product = rajni_rhos[layer] * rajni_eta
            
            # Keep ratio
            keep_ratio = rajni_product ** (-gamma)
            keep_ratio = max(0.25, min(4.0, keep_ratio))  # Clamp
            
            # Store results
            if not (exact_cls_ratio != exact_cls_ratio):
                all_results['exact_cls_ratio'][layer].append(exact_cls_ratio)
            all_results['rajni_rho'][layer].append(rajni_rhos[layer])
            all_results['exact_patch_mass'][layer].append(patch_masses[layer])
            all_results['rajni_mass'][layer].append(rajni_masses[layer])
            if exact_eta != 1.0 or prev_patch_mass is not None:
                all_results['exact_eta'][layer].append(exact_eta)
            all_results['rajni_eta'][layer].append(rajni_eta)
            if not (exact_product != exact_product):
                all_results['exact_product'][layer].append(exact_product)
            all_results['rajni_product'][layer].append(rajni_product)
            all_results['keep_ratio'][layer].append(keep_ratio)
            
            # Print
            exact_cls_str = f"{exact_cls_ratio:.4f}" if not (exact_cls_ratio != exact_cls_ratio) else "N/A"
            exact_prod_str = f"{exact_product:.4f}" if not (exact_product != exact_product) else "N/A"
            
            print(f"  {layer:^6}│{cls_grad_norms[layer]:^12.4f}│{exact_cls_str:^10}│{rajni_rhos[layer]:^10.4f}│{patch_masses[layer]:^12.4f}│{rajni_masses[layer]:^12.4f}│{exact_eta:^12.4f}│{rajni_eta:^12.4f}│{keep_ratio:^10.4f}")
            
            prev_patch_mass = patch_masses[layer + 1]
            prev_rajni_mass = rajni_masses[layer]
    
    # === Summary ===
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    
    print(f"\n  {'Layer':^6}│{'Mean Exact':^14}│{'Mean RAJNI':^14}│{'MAE':^10}│{'Mean Exact':^14}│{'Mean RAJNI':^14}│{'MAE':^10}")
    print(f"  {'':^6}│{'CLS Ratio':^14}│{'ρ':^14}│{'ρ':^10}│{'η':^14}│{'η':^14}│{'η':^10}")
    print(f"  {'─' * 6}┼{'─' * 14}┼{'─' * 14}┼{'─' * 10}┼{'─' * 14}┼{'─' * 14}┼{'─' * 10}")
    
    for layer in range(num_layers):
        # ρ statistics
        if len(all_results['exact_cls_ratio'][layer]) > 0:
            mean_exact_rho = sum(all_results['exact_cls_ratio'][layer]) / len(all_results['exact_cls_ratio'][layer])
        else:
            mean_exact_rho = float('nan')
        
        mean_rajni_rho = sum(all_results['rajni_rho'][layer]) / len(all_results['rajni_rho'][layer])
        
        if len(all_results['exact_cls_ratio'][layer]) > 0:
            mae_rho = sum(abs(e - r) for e, r in zip(all_results['exact_cls_ratio'][layer], all_results['rajni_rho'][layer])) / len(all_results['exact_cls_ratio'][layer])
        else:
            mae_rho = float('nan')
        
        # η statistics
        if len(all_results['exact_eta'][layer]) > 0:
            mean_exact_eta = sum(all_results['exact_eta'][layer]) / len(all_results['exact_eta'][layer])
        else:
            mean_exact_eta = float('nan')
        
        mean_rajni_eta = sum(all_results['rajni_eta'][layer]) / len(all_results['rajni_eta'][layer])
        
        if len(all_results['exact_eta'][layer]) > 0 and len(all_results['rajni_eta'][layer]) == len(all_results['exact_eta'][layer]):
            mae_eta = sum(abs(e - r) for e, r in zip(all_results['exact_eta'][layer], all_results['rajni_eta'][layer])) / len(all_results['exact_eta'][layer])
        else:
            mae_eta = float('nan')
        
        exact_rho_str = f"{mean_exact_rho:.4f}" if not (mean_exact_rho != mean_exact_rho) else "N/A"
        mae_rho_str = f"{mae_rho:.4f}" if not (mae_rho != mae_rho) else "N/A"
        exact_eta_str = f"{mean_exact_eta:.4f}" if not (mean_exact_eta != mean_exact_eta) else "N/A"
        mae_eta_str = f"{mae_eta:.4f}" if not (mae_eta != mae_eta) else "N/A"
        
        print(f"  {layer:^6}│{exact_rho_str:^14}│{mean_rajni_rho:^14.4f}│{mae_rho_str:^10}│{exact_eta_str:^14}│{mean_rajni_eta:^14.4f}│{mae_eta_str:^10}")
    
    # === Interpretation ===
    print(f"\n{'=' * 80}")
    print("INTERPRETATION")
    print(f"{'=' * 80}")
    print("""
RAJNI Approximation Quality:

• ρ (rho) = 1 + A(CLS→CLS) · ||V_CLS||
  Approximates the ratio of CLS sensitivity: (∂y/∂CLS_{l+1}) / (∂y/∂CLS_l)
  - ρ > 1: CLS is accumulating information (stable)
  - ρ < 1: CLS is losing information (unstable)

• η (eta) = M_l / M_{l-1}  where M = Σ importance
  Tracks patch information flow across layers
  - η > 1: Information spreading to more patches
  - η < 1: Information concentrating (good for pruning)

• keep_ratio = (ρ · η)^(-γ)
  - ρ · η > 1 → keep_ratio < 1 → prune tokens
  - ρ · η < 1 → keep_ratio > 1 → keep tokens (but clamped to 1.0 in practice)

Key Insights:
  - If MAE is small, RAJNI's approximation is accurate
  - Early layers often have η ≈ 1 (no info concentration yet)
  - Later layers should show η < 1 if pruning is beneficial
  - The "+1" in ρ accounts for residual connections
""")


def main():
    parser = argparse.ArgumentParser(description="Validate RAJNI eta and full schedule")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    validate_full_schedule(
        model_name=args.model,
        gamma=args.gamma,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        device=device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
