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
    python scripts/validate_rho.py --use_real_images --data_dir /path/to/imagenet
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import timm
from typing import List, Dict, Tuple, Optional


def compute_gradients_with_hooks(
    model: nn.Module,
    x: torch.Tensor,
    target_class: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Compute exact ∂y/∂CLS_l using torch.autograd.grad.
    
    We insert identity operations (multiply by 1) to create computation graph
    nodes that we can differentiate through.
    
    Args:
        model: timm ViT model
        x: Input images [B, 3, H, W]
        target_class: Class to compute gradient for (None = predicted class)
    
    Returns:
        cls_grads: List of gradients ∂y/∂CLS_l for each layer
        attn_values: Attention weights and values at each layer
    """
    model.eval()
    
    # Storage for captured data
    attn_values: List[Tuple[torch.Tensor, torch.Tensor]] = []
    cls_tokens: List[torch.Tensor] = []  # CLS tokens at each layer (in graph)
    
    num_heads = model.blocks[0].attn.num_heads
    head_dim = model.embed_dim // num_heads
    scale = model.blocks[0].attn.scale
    
    # Forward pass with gradient tracking
    x = x.requires_grad_(True)
    
    # Patch embedding
    h = model.patch_embed(x)
    h = model._pos_embed(h)
    h = model.patch_drop(h)
    
    for i, blk in enumerate(model.blocks):
        B, N, C = h.shape
        
        # Create a "checkpoint" for the CLS token that stays in the graph
        # We use an identity operation that creates a leaf we can differentiate
        cls_checkpoint = h[:, 0, :].clone()  # [B, D]
        cls_checkpoint.requires_grad_(True)
        cls_checkpoint.retain_grad()
        cls_tokens.append(cls_checkpoint)
        
        # Reconstruct h with the checkpoint (this keeps it in the graph)
        h_new = h.clone()
        h_new[:, 0, :] = cls_checkpoint
        h = h_new
        
        # Manual attention computation to capture weights
        x_norm = blk.norm1(h)
        
        qkv = blk.attn.qkv(x_norm)
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        
        # Store attention and values (detached for analysis)
        attn_values.append((attn.detach().clone(), v.detach().clone()))
        
        # Complete the attention with dropout
        attn_dropped = blk.attn.attn_drop(attn)
        attn_out = (attn_dropped @ v).transpose(1, 2).reshape(B, N, C)
        attn_out = blk.attn.proj(attn_out)
        attn_out = blk.attn.proj_drop(attn_out)
        
        # Residual + MLP
        h = h + blk.drop_path1(attn_out)
        h = h + blk.drop_path2(blk.mlp(blk.norm2(h)))
    
    # Final CLS token checkpoint (after all blocks)
    final_cls = h[:, 0, :].clone()
    final_cls.requires_grad_(True)
    final_cls.retain_grad()
    cls_tokens.append(final_cls)
    
    # Reconstruct for final layers
    h_final = h.clone()
    h_final[:, 0, :] = final_cls
    
    # Classification head
    h_norm = model.norm(h_final)
    logits = model.head(h_norm[:, 0])
    
    # Select target class
    if target_class is None:
        target_class = logits.argmax(dim=-1)
    
    # Create scalar loss for backward
    if isinstance(target_class, int):
        loss = logits[:, target_class].sum()
    else:
        # Gather the logits for each sample's target class
        loss = logits.gather(1, target_class.unsqueeze(1)).sum()
    
    # Backward pass
    loss.backward(retain_graph=True)
    
    # Collect gradients
    cls_grads = []
    for cls_token in cls_tokens:
        if cls_token.grad is not None:
            cls_grads.append(cls_token.grad.detach().clone())
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


def get_imagenet_val_loader(data_dir: str, batch_size: int = 4, num_workers: int = 4):
    """Create ImageNet validation loader with proper transforms."""
    from torchvision import datasets, transforms
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_dir = Path(data_dir) / "val"
    if not val_dir.exists():
        val_dir = Path(data_dir)
    
    dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return loader


def validate_rho(
    model_name: str = "vit_base_patch16_224",
    batch_size: int = 4,
    num_samples: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    use_real_images: bool = False,
    data_dir: Optional[str] = None,
    val_loader=None,  # Can pass existing loader from Kaggle
):
    """
    Main validation function.
    
    Computes and compares:
    1. Exact: ||∂y/∂CLS_l|| via backprop
    2. Exact ratio: ||∂y/∂CLS_{l+1}|| / ||∂y/∂CLS_l||
    3. RAJNI ρ: 1 + A(CLS→CLS) · ||V_CLS||
    
    Args:
        model_name: timm model name
        batch_size: Batch size for validation
        num_samples: Number of batches to test
        device: Device to run on
        seed: Random seed
        use_real_images: Use real images instead of random tensors
        data_dir: Path to ImageNet (if use_real_images and no val_loader)
        val_loader: Existing DataLoader to use (e.g., from Kaggle)
    """
    torch.manual_seed(seed)
    
    print("=" * 70)
    print("RAJNI ρ (rho) Validation")
    print("=" * 70)
    print(f"\nModel: {model_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Num samples: {num_samples}")
    print(f"Using real images: {use_real_images or val_loader is not None}")
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
    
    # Setup data source
    if val_loader is not None:
        print("Using provided val_loader")
        data_iter = iter(val_loader)
    elif use_real_images and data_dir:
        print(f"Loading ImageNet validation from {data_dir}...")
        val_loader = get_imagenet_val_loader(data_dir, batch_size=batch_size)
        data_iter = iter(val_loader)
    else:
        data_iter = None
        print("Using random input tensors (normalized like ImageNet)")
    
    # Accumulators for statistics
    all_exact_ratios = [[] for _ in range(num_layers)]
    all_rajni_rhos = [[] for _ in range(num_layers)]
    all_grad_norms = [[] for _ in range(num_layers + 1)]
    
    for sample_idx in range(num_samples):
        print(f"\n{'─' * 70}")
        print(f"Sample {sample_idx + 1}/{num_samples}")
        print(f"{'─' * 70}")
        
        # Get input
        if data_iter is not None:
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(val_loader)
                images, labels = next(data_iter)
            x = images.to(device)
            print(f"  Using real images, batch labels: {labels.tolist()[:4]}...")
        else:
            # Random normalized input (like ImageNet stats)
            x = torch.randn(batch_size, 3, 224, 224, device=device)
            # Normalize to ImageNet-like distribution
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            x = x * std + mean
        
        # Compute exact gradients using hooks
        cls_grads, attn_values = compute_gradients_with_hooks(model, x)
        
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
    
    return all_exact_ratios, all_rajni_rhos, all_grad_norms


def main():
    parser = argparse.ArgumentParser(description="Validate RAJNI rho approximation")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224",
                        help="timm model name")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for validation")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of batches to test")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use_real_images", action="store_true",
                        help="Use real ImageNet images instead of random tensors")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to ImageNet directory (required if --use_real_images)")
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.use_real_images and not args.data_dir:
        print("ERROR: --data_dir required when using --use_real_images")
        sys.exit(1)
    
    validate_rho(
        model_name=args.model,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        device=device,
        seed=args.seed,
        use_real_images=args.use_real_images,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
