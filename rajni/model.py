"""
RAJNI: Adaptive Jacobian-based Token Pruning for Vision Transformers.

Correct execution order:
CHEAP CLS ATTENTION → PRUNE → FULL ATTENTION
"""
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import logging

logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

from .pruning import (
    compute_cls_sensitivity,
    compute_jacobian_importance,
    compute_keep_ratio,
    select_tokens,
)


class AdaptiveJacobianPrunedViT(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        gamma: float = 0.01,
        min_tokens: int = 16,
        eps: float = 1e-6,
        collect_stats: bool = True,
    ):
        super().__init__()
        self.base_model = model
        self.gamma = gamma
        self.min_tokens = min_tokens
        self.eps = eps
        self.collect_stats = collect_stats

        self.embed_dim = model.embed_dim
        self.num_heads = model.blocks[0].attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads

        self._last_stats: Optional[Dict[str, Any]] = None

    def get_last_stats(self):
        return self._last_stats

    # --------------------------------------------------
    # Cheap CLS-only attention (NO full QKV)
    # --------------------------------------------------
    def _cheap_cls_attention(self, x, blk):
        """
        Computes:
        - Q only for CLS
        - K,V for all tokens
        Complexity: O(N·D)
        """
        B, N, _ = x.shape
        attn = blk.attn

        # ---- Linear projections (separated!) ----
        q = attn.q(x[:, :1])        # [B,1,D]
        k = attn.k(x)               # [B,N,D]
        v = attn.v(x)               # [B,N,D]

        # reshape to heads
        q = q.reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # CLS attention
        attn_cls = (q @ k.transpose(-2, -1)) * attn.scale
        attn_cls = attn_cls.softmax(dim=-1)  # [B,H,1,N]

        return attn_cls, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            B = x.size(0)
            m = self.base_model

            # ---- Patch + Pos ----
            x = m.patch_embed(x)
            x = m._pos_embed(x)
            x = m.patch_drop(x)

            N = x.size(1) - 1
            token_counts: List[int] = []

            prev_mass = torch.tensor(1.0, device=x.device)

            for i, blk in enumerate(m.blocks):
                token_counts.append(x.size(1))

                # ==================================================
                # 1️⃣ CHEAP IMPORTANCE (CLS-only)
                # ==================================================
                if N > self.min_tokens:
                    x_norm = blk.norm1(x)
                    attn_cls, v = self._cheap_cls_attention(x_norm, blk)

                    # Fake full attention ONLY logically (no N² compute)
                    attn_fake = attn_cls.expand(-1, -1, x.size(1), -1)

                    rho = compute_cls_sensitivity(attn_fake, v, layer_idx=i)
                    importance, mass = compute_jacobian_importance(
                        attn_fake, v, N, self.eps
                    )

                    keep_ratio = compute_keep_ratio(
                        rho, mass, prev_mass, self.gamma, self.eps
                    )

                    N_next = max(self.min_tokens, int(N * keep_ratio.item()))

                    if N_next < N:
                        keep_idx = select_tokens(importance, N_next, x.device)
                        x = torch.index_select(x, dim=1, index=keep_idx)
                        N = N_next

                    prev_mass = mass

                # ==================================================
                # 2️⃣ FULL ATTENTION (ONLY ON PRUNED TOKENS)
                # ==================================================
                x = x + blk.drop_path1(
                    blk.attn(blk.norm1(x))
                )
                x = x + blk.drop_path2(
                    blk.mlp(blk.norm2(x))
                )

            # ---- Head ----
            x = m.norm(x)
            logits = m.head(x[:, 0])

            if self.collect_stats:
                self._last_stats = {"token_counts": token_counts}
            else:
                self._last_stats = None

            return logits