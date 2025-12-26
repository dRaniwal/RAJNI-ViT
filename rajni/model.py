"""
RAJNI DEBUG VERSION
Purpose: identify why speedup is not scaling with token pruning
"""

from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import time
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
        debug: bool = True,          # 👈 DEBUG FLAG
    ):
        super().__init__()
        self.base_model = model
        self.gamma = gamma
        self.min_tokens = min_tokens
        self.eps = eps
        self.collect_stats = collect_stats
        self.debug = debug

        self.embed_dim = model.embed_dim
        self.num_heads = model.blocks[0].attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads

        self._last_stats: Optional[Dict[str, Any]] = None

    def get_last_stats(self):
        return self._last_stats

    # --------------------------------------------------
    # Cheap CLS-only attention (NO full N×N attention)
    # --------------------------------------------------
    def _cheap_cls_attention(self, x, blk):
        B, N, D = x.shape
        attn = blk.attn
        H = self.num_heads
        Dh = self.head_dim

        qkv = attn.qkv(x)
        qkv = qkv.reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_cls = q[:, :, :1]                       # [B,H,1,Dh]
        attn_cls = (q_cls @ k.transpose(-2, -1)) * attn.scale
        attn_cls = attn_cls.softmax(dim=-1)

        return attn_cls, v

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
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

            total_t_cheap = 0.0
            total_t_full = 0.0

            for i, blk in enumerate(m.blocks):
                token_counts.append(x.size(1))

                # -----------------------------
                # CHEAP IMPORTANCE
                # -----------------------------
                if N > self.min_tokens:
                    t0 = time.perf_counter()

                    x_norm = blk.norm1(x)
                    attn_cls, v = self._cheap_cls_attention(x_norm, blk)

                    # Fake attention tensor for reuse (logical only)
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
                    total_t_cheap += time.perf_counter() - t0

                # -----------------------------
                # FULL ATTENTION (EXPENSIVE)
                # -----------------------------
                t1 = time.perf_counter()

                x = x + blk.drop_path1(
                    blk.attn(blk.norm1(x))
                )
                x = x + blk.drop_path2(
                    blk.mlp(blk.norm2(x))
                )

                total_t_full += time.perf_counter() - t1

                if self.debug:
                    print(
                        f"[Layer {i:02d}] "
                        f"tokens entering full attn = {x.size(1)}"
                    )

            # ---- Head ----
            x = m.norm(x)
            logits = m.head(x[:, 0])

            if self.collect_stats:
                self._last_stats = {
                    "token_counts": token_counts,
                    "t_cheap_sec": total_t_cheap,
                    "t_full_sec": total_t_full,
                    "cheap_vs_full_ratio": total_t_cheap / (total_t_full + 1e-9),
                }

            if self.debug:
                print("\n=== TIMING SUMMARY ===")
                print(f"Cheap path time: {total_t_cheap:.4f}s")
                print(f"Full attn time : {total_t_full:.4f}s")
                print(f"Cheap / Full   : {total_t_cheap / (total_t_full + 1e-9):.3f}")

            return logits