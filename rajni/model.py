"""
RAJNI: Adaptive Jacobian-based Token Pruning for Vision Transformers.

Correct execution order:
cheap attention → prune → full attention
"""
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import logging

logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)

if hasattr(torch, "_dynamo"):
    torch._dynamo.config.capture_scalar_outputs = True

if hasattr(torch, "_inductor"):
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None

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

        self.add_module("base_model", model)
        self.gamma = gamma
        self.min_tokens = min_tokens
        self.eps = eps
        self.collect_stats = collect_stats

        self.num_heads = model.blocks[0].attn.num_heads
        self.embed_dim = model.embed_dim

        self._last_stats: Optional[Dict[str, Any]] = None

    def get_last_stats(self):
        return self._last_stats

    # --------------------------------------------------
    # Cheap CLS-only attention (NO N²)
    # --------------------------------------------------
    def _extract_qkv(self, x_norm, attn, B, N):
        head_dim = self.embed_dim // self.num_heads
        qkv = attn.qkv(x_norm).reshape(
            B, N, 3, self.num_heads, head_dim
        ).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]  # q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():

            B = x.size(0)
            m = self.base_model

            # ---- Patch + Pos ----
            x = m.patch_embed(x)
            x = m._pos_embed(x)
            x = m.patch_drop(x)

            N = x.size(1) - 1
            num_blocks = len(m.blocks)

            token_counts = [0] * num_blocks
            kept_indices_gpu = [None] * num_blocks

            prev_mass = torch.tensor(1.0, device=x.device)

            for i, blk in enumerate(m.blocks):
                token_counts[i] = x.size(1)

                # ======================================================
                # 1️⃣ CHEAP ATTENTION (CLS-only) → importance
                # ======================================================
                if N > self.min_tokens:
                    x_norm = blk.norm1(x)
                    q, k, v = self._extract_qkv(
                        x_norm, blk.attn, B, x.size(1)
                    )

                    # CLS-only attention (O(N))
                    q_cls = q[:, :, 0:1]                 # [B,H,1,D]
                    k_all = k                            # [B,H,N,D]
                    attn_cls = (q_cls @ k_all.transpose(-2, -1)) * blk.attn.scale
                    attn_cls = attn_cls.softmax(dim=-1) # [B,H,1,N]

                    # Fake full-attn tensor (ONLY for reuse of pruning code)
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
                        keep_idx = select_tokens(
                            importance, N_next, x.device
                        )

                        if self.collect_stats:
                            kept_indices_gpu[i] = keep_idx.detach()

                        x = torch.index_select(x, dim=1, index=keep_idx)
                        N = N_next

                    prev_mass = mass

                # ======================================================
                # 2️⃣ FULL ATTENTION (ONLY ON REMAINING TOKENS)
                # ======================================================
                x_norm = blk.norm1(x)
                q, k, v = self._extract_qkv(
                    x_norm, blk.attn, B, x.size(1)
                )

                attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
                attn = attn.softmax(dim=-1)

                out = (attn @ v).transpose(1, 2).reshape(B, x.size(1), -1)
                out = blk.attn.proj(out)

                x = x + blk.drop_path1(out)
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