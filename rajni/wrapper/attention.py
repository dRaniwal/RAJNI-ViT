import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .importance import compute_importance


class RAJNIAttention(nn.Module):
    """
    RAJNI Attention with dynamic D_l threshold gating.
    Optimized for A100 / modern GPUs.
    """
    __constants__ = ['num_heads', 'scale', 'head_dim', 'embed_dim',
                     'layer_idx', 'percentile', 'kr_min', 'gamma', 'tau_warmup']

    def __init__(
        self,
        attn: nn.Module,
        layer_idx: int,
        *,
        percentile: float = 0.75,
        kr_min: float = 0.60,
        gamma: float = 2.5,
        tau_warmup: float = 0.10,  # Dynamic threshold for Phase 1 vs Phase 2
    ):
        super().__init__()
        self.num_heads = attn.num_heads
        self.scale = attn.scale
        self.qkv = attn.qkv
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

        # Pre-computed constants
        self.head_dim = attn.qkv.out_features // (3 * self.num_heads)
        self.embed_dim = attn.qkv.out_features // 3

        self.layer_idx = layer_idx
        self.percentile = percentile
        self.kr_min = kr_min
        self.gamma = gamma
        self.tau_warmup = tau_warmup

        # Register buffer for CLS index
        self.register_buffer('_cls_zero', torch.tensor(0, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x)

        # EDGE CASE: only CLS token or no patches remain
        if N <= 2:
            return self._full_attn_from_qkv(qkv, B, N, C)

        num_patches = N - 1

        # ---- Importance Calculation ----
        scores = compute_importance(qkv, self.num_heads)
        patch_scores = scores[:, 1:].add(1e-12)   # [B, num_patches]
        log_scores = patch_scores.log()

        # Mean + std deviation estimation (approx 75th percentile)
        mu = log_scores.mean(dim=1, keepdim=True)
        sigma = log_scores.std(dim=1, keepdim=True)
        q_val = mu + (sigma * 0.675)

        diff = (q_val - log_scores).clamp_(min=0.0)

        # ---- Global Dispersion (D_l) ----
        D_l = (diff.mean(dim=1) / q_val.abs().squeeze(1)).mean().item()

        # ==================================================
        # DYNAMIC SKIP LAYER GATING
        # ==================================================
        if D_l < self.tau_warmup:
            # Phase 1: Network is gathering diffuse features. Skip pruning!
            return self._full_attn_from_qkv(qkv, B, N, C)

        # Phase 2: Semantic saturation achieved. Prune aggressively.
        keep_ratio = torch.exp(torch.tensor(-self.gamma * D_l)).clamp(min=self.kr_min)
        keep = int(round(keep_ratio.item() * num_patches))

        # Safety clamp
        keep = max(1, min(keep, num_patches))

        # If math says keep everything, bypass gather ops
        if keep == num_patches:
            return self._full_attn_from_qkv(qkv, B, N, C)

        # ---- Top-k (true pruning, no mask tricks) ----
        _, idx = torch.topk(patch_scores, keep, dim=1, sorted=False)

        # ---- Build keep_idx (CLS + patches) ----
        keep_idx = torch.empty(B, keep + 1, device=qkv.device, dtype=torch.long)
        keep_idx[:, 0] = 0
        keep_idx[:, 1:] = idx + 1

        # ---- Gather (actual FLOP reduction) ----
        qkv = qkv.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, qkv.size(-1)))

        Np = keep + 1
        q, k, v = self._split_qkv(qkv, B, Np)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, Np, C)
        out = self.proj_drop(self.proj(out))

        return out, keep_idx

    def _full_attn_from_qkv(self, qkv: torch.Tensor, B: int, N: int, C: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bypass pruning and process the full sequence through SDPA."""
        q, k, v = self._split_qkv(qkv, B, N)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))

        keep_idx = torch.arange(N, device=qkv.device, dtype=torch.long).expand(B, -1)
        return out, keep_idx

    def _split_qkv(self, qkv: torch.Tensor, B: int, N: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Efficient QKV split and reshape."""
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        return q, k, v
