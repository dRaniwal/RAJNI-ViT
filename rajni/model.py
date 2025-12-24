"""
RAJNI: Adaptive Jacobian-based Token Pruning for Vision Transformers.

This module provides the main model wrapper that applies inference-time
token pruning to any timm-compatible Vision Transformer.
"""
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from torch.amp import autocast

from .pruning import (
    compute_cls_sensitivity,
    compute_jacobian_importance,
    compute_keep_ratio,
    select_tokens,
)


class AdaptiveJacobianPrunedViT(nn.Module):
    """
    Wrapper for Vision Transformers with adaptive Jacobian-based pruning.
    
    This class wraps a pretrained ViT and applies layer-wise token pruning
    during inference. The pruning decisions are based on:
    
    1. Jacobian importance: Approximates d(CLS)/d(patch) using attention
       weights and value vector norms
    2. Adaptive budgeting: Adjusts pruning rate based on layer dynamics
    
    Key features:
    - Inference-only (no training required)
    - DataParallel-safe (returns only tensors from forward)
    - Framework-agnostic (works with any timm ViT)
    
    Args:
        model: A timm Vision Transformer model
        gamma: Pruning aggressiveness (default: 0.01)
        min_tokens: Minimum tokens to keep per layer (default: 16)
        eps: Numerical stability constant (default: 1e-6)
    """

    def __init__(
        self,
        model: nn.Module,
        gamma: float = 0.01,
        min_tokens: int = 16,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        
        self.m = model
        self.blocks = model.blocks
        
        self.gamma = gamma
        self.min_tokens = min_tokens
        self.eps = eps
        
        self.num_heads = self.blocks[0].attn.num_heads
        self.embed_dim = model.embed_dim
        
        self._last_stats: Optional[Dict[str, Any]] = None

    def get_last_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics from the last forward pass."""
        return self._last_stats

    def _extract_attention_values(
        self,
        x_norm: torch.Tensor,
        attn_module: nn.Module,
        batch_size: int,
        num_tokens: int,
    ) -> tuple:
        """Extract attention weights and value vectors from attention module."""
        head_dim = self.embed_dim // self.num_heads
        
        qkv = attn_module.qkv(x_norm)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = attn_module.scale
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, -1)
        out = attn_module.proj(out)
        
        return attn, v, out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive token pruning."""
        B = x.size(0)
        
        with autocast(device_type="cuda"):
            x = self.m.patch_embed(x)
            x = self.m._pos_embed(x)
            x = self.m.patch_drop(x)
        
        N = x.size(1) - 1
        
        token_counts: List[int] = []
        kept_indices: List[torch.Tensor] = []
        prev_mass: Optional[torch.Tensor] = None
        
        for blk in self.blocks:
            with autocast(device_type="cuda"):
                x_norm = blk.norm1(x)
                attn, v, attn_out = self._extract_attention_values(
                    x_norm, blk.attn, B, x.size(1)
                )
                x = x + blk.drop_path1(attn_out)
                x = x + blk.drop_path2(blk.mlp(blk.norm2(x)))
            
            token_counts.append(N + 1)
            
            if N <= self.min_tokens:
                continue
            
            rho = compute_cls_sensitivity(attn, v)
            importance, mass = compute_jacobian_importance(attn, v, N, self.eps)
            
            if prev_mass is not None:
                keep_ratio = compute_keep_ratio(rho, mass, prev_mass, self.gamma, self.eps)
                N_next = max(self.min_tokens, int(N * keep_ratio))
            else:
                N_next = N
            
            if N_next < N:
                keep_idx = select_tokens(importance, N_next, x.device)
                kept_indices.append(keep_idx.detach().cpu())
                x = x[:, keep_idx]
                N = N_next
            
            prev_mass = mass
        
        with autocast(device_type="cuda"):
            x = self.m.norm(x)
            logits = self.m.head(x[:, 0])
        
        self._last_stats = {
            "token_counts": token_counts,
            "kept_indices": kept_indices,
        }
        
        return logits

    def __repr__(self) -> str:
        return (
            f"AdaptiveJacobianPrunedViT("
            f"gamma={self.gamma}, "
            f"min_tokens={self.min_tokens}, "
            f"num_blocks={len(self.blocks)}, "
            f"embed_dim={self.embed_dim})"
        )
