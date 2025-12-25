"""
RAJNI: Adaptive Jacobian-based Token Pruning for Vision Transformers.

This module provides the main model wrapper that applies inference-time
token pruning to any timm-compatible Vision Transformer.

Performance notes:
- All operations stay on GPU to avoid CPU-GPU sync
- Indices are collected only at the end for visualization
- Compatible with torch.compile for additional speedup
"""
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import logging

# Suppress verbose torch.compile warnings for dynamic shapes
# RAJNI has variable token counts which triggers many symbolic shape warnings
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)

# Enable scalar output capture for torch.compile compatibility
# This allows .item() calls to be traced without graph breaks
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True

# Configure inductor for dynamic shapes (RAJNI has variable token counts)
# This avoids excessive CUDA graph recordings and related warnings
if hasattr(torch, '_inductor'):
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None

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
    - High GPU utilization (no CPU-GPU sync in forward pass)
    
    Args:
        model: A timm Vision Transformer model
        gamma: Pruning aggressiveness (default: 0.01)
        min_tokens: Minimum tokens to keep per layer (default: 16)
        eps: Numerical stability constant (default: 1e-6)
        collect_stats: Whether to collect pruning stats (default: True)
                       Set to False for maximum speed in production
    """

    def __init__(
        self,
        model: nn.Module,
        gamma: float = 0.01,
        min_tokens: int = 16,
        eps: float = 1e-6,
        collect_stats: bool = True,
    ) -> None:
        super().__init__()
        
        # Register the base model as a submodule using add_module
        # This ensures proper replication with DataParallel
        self.add_module('base_model', model)
        
        self.gamma = gamma
        self.min_tokens = min_tokens
        self.eps = eps
        self.collect_stats = collect_stats
        
        self.num_heads = model.blocks[0].attn.num_heads
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
        
        #CLS attn:
               
        scale = attn_module.scale
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        q_cls = q[:, :, :1]          # [B, H, 1, D]
        k_patch = k[:, :, 1:]        # [B, H, N, D]

        A_cls = (q_cls @ k_patch.transpose(-2, -1)) * attn.scale
        A_cls = A_cls.squeeze(2).mean(1)   # [B, N] 
        
        out = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, -1)
        out = attn_module.proj(out)
        
        return attn, v, out, A_cls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():

            """Forward pass with adaptive token pruning."""
            B = x.size(0)
            
            # Access base model components
            m = self.base_model
            
            # Patch embedding and positional encoding
            x = m.patch_embed(x)
            x = m._pos_embed(x)
            x = m.patch_drop(x)
            
            N = x.size(1) - 1
            
            # Pre-allocate lists (avoid dynamic allocation overhead)
            num_blocks = len(m.blocks)
            token_counts: List[int] = [0] * num_blocks
            kept_indices_gpu: List[Optional[torch.Tensor]] = [None] * num_blocks
            
            prev_mass = torch.tensor(1.0, device=x.device)            
            for i, blk in enumerate(m.blocks):
                # Record token count (this is just an int, no GPU sync)
                token_counts[i] = x.size(1)
                
                # Process through attention and MLP
                x_norm = blk.norm1(x)
                attn, v, attn_out, A_cls = self._extract_attention_values(
                    x_norm, blk.attn, B, x.size(1)
                )
                x = x + blk.drop_path1(attn_out)
                x = x + blk.drop_path2(blk.mlp(blk.norm2(x)))
                
                # Skip pruning if already at minimum
                if N <= self.min_tokens:
                    prev_mass = None
                    continue
                
                # Compute importance scores (all on GPU)
                rho = compute_cls_sensitivity(attn, v, A_cls, layer_idx=i)
                importance, mass = compute_jacobian_importance(attn, v, N, self.eps)
                
                # Adaptive keep ratio (stays on GPU, scalar captured by dynamo)
                keep_ratio = compute_keep_ratio(rho, mass, prev_mass, self.gamma, self.eps)
                N_next = max(self.min_tokens, int(N * keep_ratio.item()))
                keep_ratio = compute_keep_ratio(rho, mass, prev_mass, self.gamma, self.eps)

                if keep_ratio >= 0.999:
                    prev_mass = mass
                    continue
                # Prune tokens if needed
                if N_next < N:
                    keep_idx = select_tokens(importance, N_next, x.device)
                    
                    # Store on GPU, move to CPU only at the end
                    if self.collect_stats:
                        kept_indices_gpu[i] = keep_idx.detach()
                    
                    # Index selection (contiguous memory access)
                    x = torch.index_select(x, dim=1, index=keep_idx)
                    N = N_next
                
                prev_mass = mass
            
            # Final norm and classification head
            x = m.norm(x)
            logits = m.head(x[:, 0])
            
            # Move stats to CPU AFTER forward pass completes (batch all transfers)
            if self.collect_stats:
                kept_indices_cpu = [
                    idx.cpu() if idx is not None else None 
                    for idx in kept_indices_gpu
                ]
                self._last_stats = {
                    "token_counts": token_counts,
                    "kept_indices": [idx for idx in kept_indices_cpu if idx is not None],
                }
            else:
                self._last_stats = {"token_counts": token_counts, "kept_indices": []}
            
            return logits

    def __repr__(self) -> str:
        return (
            f"AdaptiveJacobianPrunedViT("
            f"gamma={self.gamma}, "
            f"min_tokens={self.min_tokens}, "
            f"num_blocks={len(self.base_model.blocks)}, "
            f"embed_dim={self.embed_dim})"
        )
