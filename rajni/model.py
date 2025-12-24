"""
RAJNI: Adaptive Jacobian-based Token Pruning for Vision Transformers.

This module provides the main model wrapper that applies inference-time
token pruning to any timm-compatible Vision Transformer.

Performance notes:
- All operations stay on GPU to avoid CPU-GPU sync
- Indices are collected only at the end for visualization
- Compatible with torch.compile for additional speedup
"""
from typing import Dict, List, Optional, Any, Tuple
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
    compute_importance_and_sensitivity,
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
    - Memory efficient (no tensor storage between layers)
    
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
        
        # Pre-compute constants (avoid recomputation in forward)
        self.num_heads = model.blocks[0].attn.num_heads
        self.embed_dim = model.embed_dim
        self.head_dim = self.embed_dim // self.num_heads
        self.num_blocks = len(model.blocks)
        self.scale = model.blocks[0].attn.scale
        
        self._last_stats: Optional[Dict[str, Any]] = None

    def get_last_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics from the last forward pass."""
        return self._last_stats

    def _extract_attention_and_forward(
        self,
        x: torch.Tensor,
        blk: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a transformer block and extract attention weights/values.
        
        This performs the full block forward pass while capturing the
        intermediate attention weights and value vectors needed for
        importance scoring.
        
        Args:
            x: Input tensor [B, N, D]
            blk: Transformer block module
            
        Returns:
            x_out: Output after block [B, N, D]
            attn: Attention weights [B, H, N, N]
            v: Value vectors [B, H, N, head_dim]
        """
        B, N, C = x.shape
        
        # Attention computation
        x_norm = blk.norm1(x)
        qkv = blk.attn.qkv(x_norm)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = blk.attn.attn_drop(attn)
        
        # Attention output
        attn_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        attn_out = blk.attn.proj(attn_out)
        attn_out = blk.attn.proj_drop(attn_out)
        
        # Apply residual and layer scale if present
        if hasattr(blk, 'ls1'):
            x = x + blk.drop_path1(blk.ls1(attn_out))
        else:
            x = x + blk.drop_path1(attn_out)
        
        # MLP
        if hasattr(blk, 'ls2'):
            x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
        else:
            x = x + blk.drop_path2(blk.mlp(blk.norm2(x)))
        
        return x, attn, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive token pruning."""
        B = x.size(0)
        m = self.base_model
        
        # Patch embedding and positional encoding
        x = m.patch_embed(x)
        x = m._pos_embed(x)
        x = m.patch_drop(x)
        
        N = x.size(1) - 1
        
        # Pre-allocate lists
        token_counts: List[int] = [0] * self.num_blocks
        kept_indices_gpu: List[Optional[torch.Tensor]] = [None] * self.num_blocks
        
        prev_mass: Optional[torch.Tensor] = None
        
        for i, blk in enumerate(m.blocks):
            # Record token count
            token_counts[i] = x.size(1)
            
            # Forward through block and extract attention
            x, attn, v = self._extract_attention_and_forward(x, blk)
            
            # Skip pruning if already at minimum
            if N <= self.min_tokens:
                prev_mass = None
                continue
            
            # Compute importance and sensitivity (CORE LOGIC - UNCHANGED)
            importance, mass, rho = compute_importance_and_sensitivity(attn, v, N, self.eps)
            
            # Adaptive keep ratio (CORE LOGIC - UNCHANGED)
            if prev_mass is not None:
                keep_ratio = compute_keep_ratio(rho, mass, prev_mass, self.gamma, self.eps)
                N_next = max(self.min_tokens, int(N * keep_ratio.item()))
            else:
                N_next = N
            
            # Prune tokens if needed
            if N_next < N:
                keep_idx = select_tokens(importance, N_next, x.device)
                
                if self.collect_stats:
                    kept_indices_gpu[i] = keep_idx.detach()
                
                x = torch.index_select(x, dim=1, index=keep_idx)
                N = N_next
            
            prev_mass = mass
        
        # Final norm and classification head
        x = m.norm(x)
        logits = m.head(x[:, 0])
        
        # Move stats to CPU AFTER forward pass completes
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
            f"num_blocks={self.num_blocks}, "
            f"embed_dim={self.embed_dim})"
        )
