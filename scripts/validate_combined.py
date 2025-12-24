#!/usr/bin/env python3
"""
RAJNI Complete Validation Script: ρ, η, and ρ×η

This script validates RAJNI's adaptive pruning schedule by comparing:
1. Exact: ∂y/∂CLS_l and ∂y/∂patches_l via backpropagation
2. ρ (rho): CLS sensitivity ratio approximation
3. η (eta): Patch information mass ratio
4. ρ×η: Combined pruning signal

Theory:
- ρ_l = 1 + A(CLS→CLS) · ||V_CLS||  (CLS stability)
- η_l = M_l / M_{l-1}  where M = Σ importance  (patch coherence)
- keep_ratio = (ρ · η)^(-γ)

Usage:
    # From command line
    python scripts/validate_combined.py --use_real_images --data_dir /path/to/imagenet
    
    # From Kaggle (pass val_loader directly)
    from scripts.validate_combined import validate_combined
    validate_combined(val_loader=val_loader, num_samples=10)
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Dict, Tuple, Optional


def compute_all_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target_class: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Compute exact ∂y/∂CLS_l and ∂y/∂patches_l via backpropagation.
    
    Uses checkpoint insertion to capture gradients at each layer.
    
    Returns:
        cls_grads: List of CLS gradients at each layer [B, D]
        patch_grads: List of patch gradients at each layer [B, N-1, D]
        attn_values: Attention weights and values at each layer
    """
    model.eval()
    
    attn_values: List[Tuple[torch.Tensor, torch.Tensor]] = []
    cls_tokens: List[torch.Tensor] = []
    patch_tokens: List[torch.Tensor] = []
    
    num_heads = model.blocks[0].attn.num_heads
    head_dim = model.embed_dim // num_heads
    scale = model.blocks[0].attn.scale
    
    x = x.requires_grad_(True)
    h = model.patch_embed(x)
    h = model._pos_embed(h)
    h = model.patch_drop(h)
    
    for i, blk in enumerate(model.blocks):
        B, N, C = h.shape
        
        # Create checkpoints for CLS and patches
        cls_checkpoint = h[:, 0, :].clone()
        cls_checkpoint.requires_grad_(True)
        cls_checkpoint.retain_grad()
        cls_tokens.append(cls_checkpoint)
        
        patch_checkpoint = h[:, 1:, :].clone()
        patch_checkpoint.requires_grad_(True)
        patch_checkpoint.retain_grad()
        patch_tokens.append(patch_checkpoint)
        
        # Reconstruct h with checkpoints
        h = torch.cat([cls_checkpoint.unsqueeze(1), patch_checkpoint], dim=1)
        
        # Manual attention to capture weights
        x_norm = blk.norm1(h)
        qkv = blk.attn.qkv(x_norm)
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn_values.append((attn.detach().clone(), v.detach().clone()))
        
        attn_dropped = blk.attn.attn_drop(attn)
        attn_out = (attn_dropped @ v).transpose(1, 2).reshape(B, N, C)
        attn_out = blk.attn.proj(attn_out)
        attn_out = blk.attn.proj_drop(attn_out)
        
        h = h + blk.drop_path1(attn_out)
        h = h + blk.drop_path2(blk.mlp(blk.norm2(h)))
    
    # Final checkpoints
    final_cls = h[:, 0, :].clone()
    final_cls.requires_grad_(True)
    final_cls.retain_grad()
    cls_tokens.append(final_cls)
    
    final_patches = h[:, 1:, :].clone()
    final_patches.requires_grad_(True)
    final_patches.retain_grad()
    patch_tokens.append(final_patches)
    
    h = torch.cat([final_cls.unsqueeze(1), final_patches], dim=1)
    
    # Classification head
    h_norm = model.norm(h)
    logits = model.head(h_norm[:, 0])
    
    if target_class is None:
        target_class = logits.argmax(dim=-1)
    
    if isinstance(target_class, int):
        loss = logits[:, target_class].sum()
    else:
        loss = logits.gather(1, target_class.unsqueeze(1)).sum()
    
    loss.backward()
    
    # Collect gradients
    cls_grads = [ct.grad.detach().clone() if ct.grad is not None else None for ct in cls_tokens]
    patch_grads = [pt.grad.detach().clone() if pt.grad is not None else None for pt in patch_tokens]
    
    return cls_grads, patch_grads, attn_values


def compute_rajni_rho(attn: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Compute RAJNI's ρ approximation: 1 + A(CLS→CLS) · ||V_CLS - mean(V)||
    
    Uses centered V (like patch importance) for consistency.
    
    Args:
        attn: [B, H, N, N]
        values: [B, H, N, D]
    
    Returns:
        rho: [B]
    """
    A_mean = attn.mean(dim=1)  # [B, N, N]
    A_cls_cls = A_mean[:, 0, 0]  # [B]
    
    V_mean = values.mean(dim=1)  # [B, N, D]
    V_cls = V_mean[:, 0]  # [B, D]
    
    # Center V_cls by subtracting mean across all tokens
    V_all_mean = V_mean.mean(dim=1)  # [B, D]
    V_cls_centered = V_cls - V_all_mean  # [B, D]
    V_cls_norm = V_cls_centered.norm(dim=-1)  # [B]
    
    return 1.0 + A_cls_cls * V_cls_norm


def compute_rajni_importance_mass(
    attn: torch.Tensor,
    values: torch.Tensor,
    num_patches: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RAJNI's importance scores and mass.
    
    importance = A(CLS→patch) · ReLU(standardized ||V_patch - mean(V)||)
    mass = sum(importance)
    
    Returns:
        importance: [B, num_patches]
        mass: [B]
    """
    A_mean = attn.mean(dim=1)  # [B, N, N]
    A_cls_to_patches = A_mean[:, 0, 1:num_patches + 1]  # [B, num_patches]
    
    V_mean = values.mean(dim=1)  # [B, N, D]
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
    
    return importance, mass


def get_imagenet_val_loader(data_dir: str, batch_size: int = 4, num_workers: int = 4):
    """Create ImageNet validation loader."""
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


def validate_combined(
    model_name: str = "vit_base_patch16_224",
    batch_size: int = 4,
    num_samples: int = 10,
    gamma: float = 0.05,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    use_real_images: bool = False,
    data_dir: Optional[str] = None,
    val_loader=None,
):
    """
    Main validation function comparing ρ, η, and ρ×η.
    
    Args:
        model_name: timm model name
        batch_size: Batch size
        num_samples: Number of batches to test
        gamma: Pruning aggressiveness parameter
        device: Device to run on
        seed: Random seed
        use_real_images: Use real images
        data_dir: Path to ImageNet
        val_loader: Existing DataLoader (e.g., from Kaggle)
    """
    torch.manual_seed(seed)
    
    print("=" * 90)
    print("RAJNI Combined Validation: ρ (rho), η (eta), and ρ×η")
    print("=" * 90)
    print(f"\nModel: {model_name}")
    print(f"Device: {device}")
    print(f"Gamma: {gamma}")
    print(f"Num samples: {num_samples}")
    print()
    
    # Load model
    print("Loading model...")
    model = timm.create_model(model_name, pretrained=True).to(device).eval()
    
    num_layers = len(model.blocks)
    num_patches = (224 // 16) ** 2  # 196 for ViT-B/16
    print(f"Layers: {num_layers}, Patches: {num_patches}, Embed dim: {model.embed_dim}")
    print()
    
    # Setup data
    if val_loader is not None:
        print("Using provided val_loader")
        data_iter = iter(val_loader)
    elif use_real_images and data_dir:
        print(f"Loading from {data_dir}...")
        val_loader = get_imagenet_val_loader(data_dir, batch_size=batch_size)
        data_iter = iter(val_loader)
    else:
        data_iter = None
        print("Using random tensors")
    
    # Accumulators
    stats = {
        'exact_cls_ratio': [[] for _ in range(num_layers)],
        'exact_patch_ratio': [[] for _ in range(num_layers)],
        'rajni_rho': [[] for _ in range(num_layers)],
        'rajni_eta': [[] for _ in range(num_layers)],
        'rajni_product': [[] for _ in range(num_layers)],
        'cls_grad_norms': [[] for _ in range(num_layers + 1)],
        'patch_grad_masses': [[] for _ in range(num_layers + 1)],
    }
    
    for sample_idx in range(num_samples):
        print(f"\n{'─' * 90}")
        print(f"Sample {sample_idx + 1}/{num_samples}")
        print(f"{'─' * 90}")
        
        # Get input
        if data_iter is not None:
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(val_loader)
                images, labels = next(data_iter)
            x = images.to(device)
            print(f"  Labels: {labels.tolist()[:4]}...")
        else:
            x = torch.randn(batch_size, 3, 224, 224, device=device)
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            x = x * std + mean
        
        # Compute exact gradients
        cls_grads, patch_grads, attn_values = compute_all_gradients(model, x)
        
        # Compute gradient norms
        cls_grad_norms = []
        patch_grad_masses = []
        
        for i, (cg, pg) in enumerate(zip(cls_grads, patch_grads)):
            if cg is not None:
                cls_norm = cg.norm(dim=-1).mean().item()
            else:
                cls_norm = 0.0
            cls_grad_norms.append(cls_norm)
            stats['cls_grad_norms'][i].append(cls_norm)
            
            if pg is not None:
                # Patch mass = sum of gradient norms across patches
                patch_mass = pg.norm(dim=-1).sum(dim=1).mean().item()
            else:
                patch_mass = 0.0
            patch_grad_masses.append(patch_mass)
            stats['patch_grad_masses'][i].append(patch_mass)
        
        # Compute RAJNI metrics
        rajni_rhos = []
        rajni_masses = []
        
        for layer in range(num_layers):
            attn, v = attn_values[layer]
            
            rho = compute_rajni_rho(attn, v).mean().item()
            rajni_rhos.append(rho)
            
            _, mass = compute_rajni_importance_mass(attn, v, num_patches)
            rajni_masses.append(mass.mean().item())
        
        # Print table header
        print(f"\n  {'Lyr':<4}│{'||∂y/∂CLS||':<12}│{'CLS Ratio':<11}│{'ρ':<8}│{'Δρ':<8}│{'Patch Mass':<12}│{'η':<8}│{'ρ×η':<8}│{'keep_r':<8}")
        print(f"  {'─'*4}┼{'─'*12}┼{'─'*11}┼{'─'*8}┼{'─'*8}┼{'─'*12}┼{'─'*8}┼{'─'*8}┼{'─'*8}")
        
        prev_rajni_mass = None
        
        for layer in range(num_layers):
            # Exact CLS ratio
            if cls_grad_norms[layer] > 1e-10:
                exact_cls_ratio = cls_grad_norms[layer + 1] / cls_grad_norms[layer]
            else:
                exact_cls_ratio = float('nan')
            
            # Exact patch ratio (η approximation target)
            if patch_grad_masses[layer] > 1e-10:
                exact_patch_ratio = patch_grad_masses[layer + 1] / patch_grad_masses[layer]
            else:
                exact_patch_ratio = float('nan')
            
            # RAJNI metrics
            rho = rajni_rhos[layer]
            
            if prev_rajni_mass is not None and prev_rajni_mass > 1e-10:
                eta = rajni_masses[layer] / prev_rajni_mass
            else:
                eta = 1.0
            
            rho_eta = rho * eta
            keep_ratio = max(0.25, min(4.0, rho_eta ** (-gamma)))
            
            # Errors
            rho_error = abs(rho - exact_cls_ratio) if exact_cls_ratio == exact_cls_ratio else float('nan')
            
            # Store stats
            if exact_cls_ratio == exact_cls_ratio:
                stats['exact_cls_ratio'][layer].append(exact_cls_ratio)
            if exact_patch_ratio == exact_patch_ratio:
                stats['exact_patch_ratio'][layer].append(exact_patch_ratio)
            stats['rajni_rho'][layer].append(rho)
            stats['rajni_eta'][layer].append(eta)
            stats['rajni_product'][layer].append(rho_eta)
            
            # Format strings
            cls_ratio_str = f"{exact_cls_ratio:.4f}" if exact_cls_ratio == exact_cls_ratio else "nan"
            rho_err_str = f"{rho_error:.4f}" if rho_error == rho_error else "N/A"
            
            print(f"  {layer:<4}│{cls_grad_norms[layer]:<12.4f}│{cls_ratio_str:<11}│{rho:<8.4f}│{rho_err_str:<8}│{patch_grad_masses[layer]:<12.4f}│{eta:<8.4f}│{rho_eta:<8.4f}│{keep_ratio:<8.4f}")
            
            prev_rajni_mass = rajni_masses[layer]
        
        # Output row
        print(f"  {'out':<4}│{cls_grad_norms[num_layers]:<12.4f}│{'─':<11}│{'─':<8}│{'─':<8}│{patch_grad_masses[num_layers]:<12.4f}│{'─':<8}│{'─':<8}│{'─':<8}")
    
    # ============== SUMMARY ==============
    print(f"\n{'=' * 90}")
    print("SUMMARY STATISTICS (averaged across all samples)")
    print(f"{'=' * 90}")
    
    print(f"\n  {'Lyr':<4}│{'Mean CLS':<12}│{'Mean':<8}│{'MAE':<8}│{'Mean':<8}│{'Mean':<8}│{'Mean':<10}│{'Implied':<10}")
    print(f"  {'':4}│{'Ratio':<12}│{'ρ':<8}│{'ρ':<8}│{'η':<8}│{'ρ×η':<8}│{'keep_ratio':<10}│{'prune %':<10}")
    print(f"  {'─'*4}┼{'─'*12}┼{'─'*8}┼{'─'*8}┼{'─'*8}┼{'─'*8}┼{'─'*10}┼{'─'*10}")
    
    total_mae_rho = 0
    valid_layers = 0
    
    for layer in range(num_layers):
        exact_ratios = stats['exact_cls_ratio'][layer]
        rhos = stats['rajni_rho'][layer]
        etas = stats['rajni_eta'][layer]
        products = stats['rajni_product'][layer]
        
        mean_exact = sum(exact_ratios) / len(exact_ratios) if exact_ratios else float('nan')
        mean_rho = sum(rhos) / len(rhos) if rhos else 0
        mean_eta = sum(etas) / len(etas) if etas else 1
        mean_product = sum(products) / len(products) if products else 1
        
        if exact_ratios and rhos:
            mae_rho = sum(abs(e - r) for e, r in zip(exact_ratios, rhos)) / len(exact_ratios)
            total_mae_rho += mae_rho
            valid_layers += 1
        else:
            mae_rho = float('nan')
        
        mean_keep = max(0.25, min(4.0, mean_product ** (-gamma)))
        prune_pct = (1 - mean_keep) * 100 if mean_keep < 1 else 0
        
        exact_str = f"{mean_exact:.4f}" if mean_exact == mean_exact else "N/A"
        mae_str = f"{mae_rho:.4f}" if mae_rho == mae_rho else "N/A"
        
        print(f"  {layer:<4}│{exact_str:<12}│{mean_rho:<8.4f}│{mae_str:<8}│{mean_eta:<8.4f}│{mean_product:<8.4f}│{mean_keep:<10.4f}│{prune_pct:<10.1f}%")
    
    print(f"\n{'─' * 90}")
    if valid_layers > 0:
        avg_mae = total_mae_rho / valid_layers
        print(f"Average MAE for ρ: {avg_mae:.4f}")
    
    # ============== GRADIENT FLOW ==============
    print(f"\n{'=' * 90}")
    print("GRADIENT FLOW ANALYSIS")
    print(f"{'=' * 90}")
    
    print("\nCLS Gradient Norms (||∂y/∂CLS_l||):")
    max_cls = max(sum(n)/len(n) for n in stats['cls_grad_norms'] if n)
    for layer in range(num_layers + 1):
        if stats['cls_grad_norms'][layer]:
            mean_norm = sum(stats['cls_grad_norms'][layer]) / len(stats['cls_grad_norms'][layer])
            bar_len = int(40 * mean_norm / max_cls) if max_cls > 0 else 0
            label = f"Layer {layer}" if layer < num_layers else "Output"
            print(f"  {label:10} │ {mean_norm:8.4f} │ {'█' * bar_len}")
    
    print("\nPatch Gradient Mass (Σ||∂y/∂patch_t||):")
    max_patch = max(sum(n)/len(n) for n in stats['patch_grad_masses'] if n)
    for layer in range(num_layers + 1):
        if stats['patch_grad_masses'][layer]:
            mean_mass = sum(stats['patch_grad_masses'][layer]) / len(stats['patch_grad_masses'][layer])
            bar_len = int(40 * mean_mass / max_patch) if max_patch > 0 else 0
            label = f"Layer {layer}" if layer < num_layers else "Output"
            print(f"  {label:10} │ {mean_mass:8.4f} │ {'█' * bar_len}")
    
    # ============== INTERPRETATION ==============
    print(f"\n{'=' * 90}")
    print("INTERPRETATION")
    print(f"{'=' * 90}")
    print(f"""
RAJNI Pruning Schedule Analysis:

• ρ (rho) = 1 + A(CLS→CLS) · ||V_CLS||
  - Approximates CLS sensitivity ratio: ||∂y/∂CLS_{{l+1}}|| / ||∂y/∂CLS_l||
  - MAE shows approximation quality (lower is better)

• η (eta) = M_l / M_{{l-1}}  where M = Σ importance
  - Tracks patch information concentration
  - η < 1 means information concentrating → good for pruning
  - η > 1 means information spreading → be conservative

• ρ × η = Combined signal
  - ρ×η > 1 → stable layer, can prune (keep_ratio < 1)
  - ρ×η < 1 → unstable layer, be conservative (keep_ratio ≥ 1)

• keep_ratio = (ρ × η)^(-γ)  with γ = {gamma}
  - Implied prune % shows how much would be pruned at each layer

Key Observations:
  - Early layers: ρ >> exact ratio (approximation is loose)
  - Later layers: ρ ≈ exact ratio (approximation tightens)
  - The "+1" in ρ accounts for residual connections
  - η modulates ρ based on patch information dynamics
""")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate RAJNI ρ, η, and ρ×η")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_real_images", action="store_true")
    parser.add_argument("--data_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.use_real_images and not args.data_dir:
        print("ERROR: --data_dir required with --use_real_images")
        sys.exit(1)
    
    validate_combined(
        model_name=args.model,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        gamma=args.gamma,
        device=device,
        seed=args.seed,
        use_real_images=args.use_real_images,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
