"""
RAJNI — Fixed Scheduled Token Pruning (Stable & Fast)

Implements SAINT-style pruning with:
- fixed r_max, alpha schedule
- importance-based token selection
- early pruning BEFORE attention
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn

from .pruning import (
    compute_jacobian_importance,
    select_tokens,
)

class AdaptiveJacobianPrunedViT(nn.Module):
    """
    Fixed-schedule token pruning for ViT.

    N_l = N_0 * (1 - r_max * (l / (L-1))^alpha)
    """

    def __init__(
        self,
        model: nn.Module,
        r_max: float = 0.6,
        alpha: float = 2.0,
        min_tokens: int = 16,
        eps: float = 1e-6,
        collect_stats: bool = True,
    ):
        super().__init__()
        self.add_module("base_model", model)

        self.r_max = r_max
        self.alpha = alpha
        self.min_tokens = min_tokens
        self.eps = eps
        self.collect_stats = collect_stats

        self.num_layers = len(model.blocks)
        self._last_stats: Optional[Dict[str, Any]] = None

    # --------------------------------------------------
    def get_last_stats(self):
        return self._last_stats

    # --------------------------------------------------
    def _target_tokens(self, layer_idx: int, N0: int) -> int:
        """Fixed pruning schedule"""
        frac = layer_idx / max(self.num_layers - 1, 1)
        keep_ratio = 1.0 - self.r_max * (frac ** self.alpha)
        keep_ratio = max(keep_ratio, 0.0)
        return max(self.min_tokens, int(N0 * keep_ratio))

    # --------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            m = self.base_model
            B = x.size(0)

            # ---- Patch + Pos ----
            x = m.patch_embed(x)
            x = m._pos_embed(x)
            x = m.patch_drop(x)

            N = x.size(1) - 1  # patches only
            N0 = N

            token_counts: List[int] = []
            kept_indices_gpu: List[Optional[torch.Tensor]] = [None] * self.num_layers

            for layer_idx, blk in enumerate(m.blocks):

                token_counts.append(x.size(1))

                # -----------------------------------------
                # PRUNE BEFORE ATTENTION (scheduled)
                # -----------------------------------------
                target_N = self._target_tokens(layer_idx, N0)

                if N > target_N:
                    # ---- cheap importance calc ----
                    x_norm = blk.norm1(x)
                    attn = blk.attn

                    qkv = attn.qkv(x_norm).reshape(
                        B, x.size(1), 3, attn.num_heads, -1
                    ).permute(2, 0, 3, 1, 4)

                    q, k, v = qkv[0], qkv[1], qkv[2]

                    A = (q @ k.transpose(-2, -1)) * attn.scale
                    A = A.softmax(dim=-1)

                    importance, _ = compute_jacobian_importance(
                        A, v, N, eps=self.eps
                    )

                    keep_idx = select_tokens(
                        importance, target_N, device=x.device
                    )

                    if self.collect_stats:
                        kept_indices_gpu[layer_idx] = keep_idx.detach()

                    x = torch.index_select(x, dim=1, index=keep_idx)
                    N = target_N

                # -----------------------------------------
                # FULL BLOCK FORWARD
                # -----------------------------------------
                x = x + blk.drop_path1(blk.attn(blk.norm1(x)))
                x = x + blk.drop_path2(blk.mlp(blk.norm2(x)))

            # ---- Head ----
            x = m.norm(x)
            logits = m.head(x[:, 0])

            if self.collect_stats:
                self._last_stats = {
                    "token_counts": token_counts,
                    "kept_indices": [
                        idx.cpu() for idx in kept_indices_gpu if idx is not None
                    ],
                }
            else:
                self._last_stats = {"token_counts": token_counts}

            return logits

    def __repr__(self):
        return (
            f"RAJNI_FixedSchedule("
            f"r_max={self.r_max}, alpha={self.alpha}, "
            f"min_tokens={self.min_tokens})"
        )