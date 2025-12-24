#!/usr/bin/env python3
"""
RAJNI ρ (rho) Validation Script

This script validates RAJNI's approximation of CLS sensitivity (ρ) by comparing:
1. Exact gradient: ∂y/∂CLS_l computed via backpropagation
2. Exact ratio: (∂y/∂CLS_{l+1}) / (∂y/∂CLS_l)  
3. RAJNI approximation: ρ_l = 1 + A(CLS→CLS) · ||V_CLS||

Theory:
- ρ approximates how CLS token sensitivity changes across layers
- Using chain rule: ∂y/∂CLS_{l+1} / ∂y/∂CLS_l ≈ ∂CLS_l/∂CLS_{l+1}
- CLS update: CLS_{l+1} ≈ A(CLS→CLS) · V_CLS + residual
- So: ∂CLS_{l+1}/∂CLS_l ≈ A(CLS→CLS) · ||V_CLS||
- With residual: ρ = 1 + A(CLS→CLS) · ||V_CLS||

Usage:
    python scripts/validate_rho.py
    python scripts/validate_rho.py --model vit_small_patch16_224
    python scripts/validate_rho.py --batch_size 4 --num_samples 10
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import timm
from typing import List, Dict, Tuple, Optional


def get_cls_tokens_with_grad(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Forward pass that captures CLS token at each layer with gradients enabled.
    
    Args:
        model: timm ViT model
        x: Input images [B, 3, H, W]
    
    Returns:
        logits: Model output [B, num_classes]
        cls_tokens: List of CLS tokens at each layer [B, D]
        attn_values: List of (attention_weights, values) at each layer
    """
    # Patch embedding
    x = model.patch_embed(x)
    x = model._pos_embed(x)
    x = model.patch_drop(x)
    
    cls_tokens = []
    attn_values = []
    
    num_heads = model.blocks[0].attn.num_heads
    head_dim = model.embed_dim // num_heads
    scale = model.blocks[0].attn.scale
    
    for i, blk in enumerate(model.blocks):
        # Store CLS token BEFORE this block (requires grad for backprop)
        cls_token = x[:, 0].clone()
        cls_token.retain_grad()
        cls_tokens.append(cls_token)
        
        # Manual attention to capture weights
        B, N, C = x.shape
        x_norm = blk.norm1(x)
        
        qkv = blk.attn.qkv(x_norm)
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        
        # Store attention and values
        attn_values.append((attn.detach().clone(), v.detach().clone()))
        
        # Complete attention
        attn_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        attn_out = blk.attn.proj(attn_out)
        attn_out = blk.attn.proj_drop(attn_out)
        
        # Residual + MLP
        x = x + blk.drop_path1(attn_out)
        x = x + blk.drop_path2(blk.mlp(blk.norm2(x)))
    
    # Final CLS token (after last block)
    cls_token = x[:, 0].clone()
    cls_token.retain_grad()
    cls_tokens.append(cls_token)
    
    # Classification head
    x = model.norm(x)
    logits = model.head(x[:, 0])
    
    return logits, cls_tokens, attn_values


def compute_exact_cls_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target_class: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Compute exact ∂y/∂CLS_l for each layer via backpropagation.
    
    Args:
        model: timm ViT model
        x: Input images [B, 3, H, W]
        target_class: Class to compute gradient for (None = predicted class)
    
    Returns:
        cls_grads: List of gradient norms ||∂y/∂CLS_l|| for each layer
        attn_values: Attention weights and values at each layer
    """
    model.eval()
    x = x.clone().requires_grad_(True)
    
    # Forward with CLS token capture
    logits, cls_tokens, attn_values = get_cls_tokens_with_grad(model, x)
    
    # Select target class
    if target_class is None:
        target_class = logits.argmax(dim=-1)
    
    # Create one-hot target
    if isinstance(target_class, int):
        target = torch.zeros_like(logits)
        target[:, target_class] = 1.0
    else:
        target = torch.zeros_like(logits)
        target.scatter_(1, target_class.unsqueeze(1), 1.0)
    
    # Compute y = sum of logits for target class (scalar for gradient)
    y = (logits * target).sum()
    
    # Backpropagate
    y.backward(retain_graph=True)
    
    # Collect gradients
    cls_grads = []
    for i, cls_token in enumerate(cls_tokens):
        if cls_token.grad is not None:
            # Gradient: ∂y/∂CLS_l [B, D]
            grad = cls_token.grad.clone()
            cls_grads.append(grad)
        else:
            cls_grads.append(None)
    
    return cls_grads, attn_values


def compute_rajni_rho(
    attn: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """
    Compute RAJNI's ρ approximation: 1 + A(CLS→CLS) · ||V_CLS||
    
    Args:
        attn: Attention weights [B, H, N, N]
        values: Value vectors [B, H, N, D]
    
    Returns:
        rho: Scalar approximation
    """
    # Average across heads
    A_mean = attn.mean(dim=1)  # [B, N, N]
    A_cls_cls = A_mean[:, 0, 0]  # [B] - CLS self-attention
    
    # CLS value vector norm (averaged across heads)
    V_cls = values.mean(dim=1)[:, 0]  # [B, D]
    V_cls_norm = V_cls.norm(dim=-1)   # [B]
    
    # RAJNI approximation
    rho = 1.0 + A_cls_cls * V_cls_norm
    
    return rho


def validate_rho(
    model_name: str = "vit_base_patch16_224",
    batch_size: int = 2,
    num_samples: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
):
    """
    Main validation function.
    
    Computes and compares:
    1. Exact: ||∂y/∂CLS_l|| via backprop
    2. Exact ratio: ||∂y/∂CLS_{l+1}|| / ||∂y/∂CLS_l||
    3. RAJNI ρ: 1 + A(CLS→CLS) · ||V_CLS||
    """
    torch.manual_seed(seed)
    
    print("=" * 70)
    print("RAJNI ρ (rho) Validation")
    print("=" * 70)
    print(f"\nModel: {model_name}")
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
    print(f"Number of layers: {num_layers}")
    print(f"Embed dim: {model.embed_dim}")
    print()
    
    # Accumulators for statistics
    all_exact_ratios = [[] for _ in range(num_layers)]
    all_rajni_rhos = [[] for _ in range(num_layers)]
    all_grad_norms = [[] for _ in range(num_layers + 1)]
    
    for sample_idx in range(num_samples):
        print(f"\n{'─' * 70}")
        print(f"Sample {sample_idx + 1}/{num_samples}")
        print(f"{'─' * 70}")
        
        # Random input
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        
        # Compute exact gradients
        cls_grads, attn_values = compute_exact_cls_gradients(model, x)
        
        # Compute gradient norms
        grad_norms = []
        for i, grad in enumerate(cls_grads):
            if grad is not None:
                norm = grad.norm(dim=-1).mean().item()  # Mean across batch
                grad_norms.append(norm)
                all_grad_norms[i].append(norm)
            else:
                grad_norms.append(0.0)
        
        print(f"\n  Layer  │  ||∂y/∂CLS_l||  │  Exact Ratio  │  RAJNI ρ  │  Δ (error)")
        print(f"  {'─' * 6}┼{'─' * 17}┼{'─' * 15}┼{'─' * 11}┼{'─' * 12}")
        
        for layer in range(num_layers):
            # Exact ratio: ||∂y/∂CLS_{l+1}|| / ||∂y/∂CLS_l||
            if grad_norms[layer] > 1e-10:
                exact_ratio = grad_norms[layer + 1] / grad_norms[layer]
            else:
                exact_ratio = float('nan')
            
            # RAJNI ρ
            attn, v = attn_values[layer]
            rajni_rho = compute_rajni_rho(attn, v).mean().item()
            
            # Store for statistics
            if not (exact_ratio != exact_ratio):  # Check for NaN
                all_exact_ratios[layer].append(exact_ratio)
            all_rajni_rhos[layer].append(rajni_rho)
            
            # Error
            if not (exact_ratio != exact_ratio):
                error = abs(rajni_rho - exact_ratio)
                error_str = f"{error:.4f}"
            else:
                error_str = "N/A"
            
            print(f"  {layer:^6}│  {grad_norms[layer]:^15.6f}│  {exact_ratio:^13.4f}│  {rajni_rho:^9.4f}│  {error_str:^10}")
        
        # Final layer (output)
        print(f"  {num_layers:^6}│  {grad_norms[num_layers]:^15.6f}│  {'(output)':^13}│  {'-':^9}│  {'-':^10}")
    
    # Summary statistics
    print(f"\n{'=' * 70}")
    print("SUMMARY STATISTICS (averaged across all samples)")
    print(f"{'=' * 70}")
    
    print(f"\n  Layer  │  Mean Exact Ratio  │  Mean RAJNI ρ  │  Mean Abs Error  │  Correlation")
    print(f"  {'─' * 6}┼{'─' * 20}┼{'─' * 16}┼{'─' * 18}┼{'─' * 14}")
    
    total_mae = 0
    valid_layers = 0
    
    for layer in range(num_layers):
        exact_ratios = all_exact_ratios[layer]
        rajni_rhos = all_rajni_rhos[layer]
        
        if len(exact_ratios) > 0:
            mean_exact = sum(exact_ratios) / len(exact_ratios)
            mean_rajni = sum(rajni_rhos) / len(rajni_rhos)
            
            # Mean absolute error
            mae = sum(abs(e - r) for e, r in zip(exact_ratios, rajni_rhos)) / len(exact_ratios)
            total_mae += mae
            valid_layers += 1
            
            # Simple correlation
            if len(exact_ratios) > 1:
                mean_e = mean_exact
                mean_r = mean_rajni
                num = sum((e - mean_e) * (r - mean_r) for e, r in zip(exact_ratios, rajni_rhos))
                den_e = sum((e - mean_e) ** 2 for e in exact_ratios) ** 0.5
                den_r = sum((r - mean_r) ** 2 for r in rajni_rhos) ** 0.5
                if den_e > 1e-10 and den_r > 1e-10:
                    corr = num / (den_e * den_r)
                else:
                    corr = float('nan')
            else:
                corr = float('nan')
            
            corr_str = f"{corr:.4f}" if not (corr != corr) else "N/A"
            print(f"  {layer:^6}│  {mean_exact:^18.4f}│  {mean_rajni:^14.4f}│  {mae:^16.4f}│  {corr_str:^12}")
    
    # Overall metrics
    print(f"\n{'─' * 70}")
    if valid_layers > 0:
        avg_mae = total_mae / valid_layers
        print(f"Average MAE across layers: {avg_mae:.4f}")
    
    # Gradient flow analysis
    print(f"\n{'=' * 70}")
    print("GRADIENT FLOW ANALYSIS")
    print(f"{'=' * 70}")
    
    print("\nMean ||∂y/∂CLS_l|| at each layer:")
    for layer in range(num_layers + 1):
        if len(all_grad_norms[layer]) > 0:
            mean_norm = sum(all_grad_norms[layer]) / len(all_grad_norms[layer])
            bar = "█" * int(min(50, mean_norm * 10))
            layer_name = f"Layer {layer}" if layer < num_layers else "Output"
            print(f"  {layer_name:10} │ {mean_norm:10.4f} │ {bar}")
    
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}")
    print("""
• Exact Ratio = ||∂y/∂CLS_{l+1}|| / ||∂y/∂CLS_l||
  This is the true sensitivity change between layers.

• RAJNI ρ = 1 + A(CLS→CLS) · ||V_CLS||
  This approximates the exact ratio using attention weights.

• If Exact Ratio ≈ RAJNI ρ, the approximation is valid.
• Larger ρ means CLS is more stable → less aggressive pruning.
• Smaller ρ means CLS is changing rapidly → more conservative pruning.

• The "+1" term accounts for the residual connection:
  CLS_{l+1} = CLS_l + Attention(CLS_l)
  So ∂CLS_{l+1}/∂CLS_l = 1 + ∂Attention/∂CLS_l
""")


def main():
    parser = argparse.ArgumentParser(description="Validate RAJNI rho approximation")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224",
                        help="timm model name")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for validation")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of random samples to test")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    validate_rho(
        model_name=args.model,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        device=device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
