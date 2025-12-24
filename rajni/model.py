"""
RAJNI: Adaptive Jacobian-based Token Pruning for Vision Transformers.

This module provides the main model wrapper that applies inference-time
token pruning to any timm-compatible Vision Transformer.

Performance notes:
- Uses native attention with patched forward to capture weights (no redundant computation)
- All operations stay on GPU to avoid CPU-GPU sync
- Indices are collected only at the end for visualization
- Compatible with torch.compile for additional speedup
"""
from typing import Dict, List, Optional, Any, Callable
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


def _make_attn_forward_with_capture(
    attn_module: nn.Module,
    storage: Dict[str, Optional[torch.Tensor]],
) -> Callable:
    """
    Create a patched attention forward that captures attention weights and values.
    
    This replaces timm's Attention.forward to store intermediate tensors
    needed for importance scoring, while maintaining identical computation.
    
    Args:
        attn_module: The timm Attention module to patch
        storage: Dict to store {'attn': tensor, 'v': tensor}
    
    Returns:
        New forward function with capture capability
    """
    num_heads = attn_module.num_heads
    scale = attn_module.scale
    
    def forward_with_capture(x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        head_dim = C // num_heads
        
        # QKV projection (same as timm)
        qkv = attn_module.qkv(x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Attention computation (same as timm)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = attn_module.attn_drop(attn)
        
        # Store for RAJNI importance scoring
        storage['attn'] = attn
        storage['v'] = v
        
        # Output projection (same as timm)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_module.proj(x)
        x = attn_module.proj_drop(x)
        
        return x
    
    return forward_with_capture


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
    - Native attention capture (no redundant computation)
    - DataParallel-safe (returns only tensors from forward)
    - Framework-agnostic (works with any timm ViT)
    
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
        
        # Storage for captured attention weights and values (per-layer)
        # Each block gets its own storage dict
        self._attn_storage: List[Dict[str, Optional[torch.Tensor]]] = [
            {'attn': None, 'v': None} for _ in range(self.num_blocks)
        ]
        
        # Patch attention modules to capture weights during forward
        self._patch_attention_modules()

    def _patch_attention_modules(self) -> None:
        """
        Patch each block's attention module to capture attention weights.
        
        This replaces the forward method with one that stores attn/v
        in self._attn_storage during computation.
        """
        for i, blk in enumerate(self.base_model.blocks):
            attn = blk.attn
            # Store original forward for potential restoration
            attn._original_forward = attn.forward
            # Replace with capturing forward
            attn.forward = _make_attn_forward_with_capture(attn, self._attn_storage[i])

    def restore_attention_modules(self) -> None:
        """
        Restore original attention forward methods.
        
        Call this if you want to use the base model without RAJNI.
        """
        for blk in self.base_model.blocks:
            attn = blk.attn
            if hasattr(attn, '_original_forward'):
                attn.forward = attn._original_forward
                delattr(attn, '_original_forward')

    def get_last_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics from the last forward pass."""
        return self._last_stats

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
            
            # Run native block forward (attention captures attn/v internally)
            x = blk(x)
            
            # Retrieve captured attention weights and values
            attn = self._attn_storage[i]['attn']
            v = self._attn_storage[i]['v']
            
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
