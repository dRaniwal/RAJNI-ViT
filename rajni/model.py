import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

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
        gamma: float = 0.5,
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

    # -------------------------
    # Cheap CLS-only attention
    # -------------------------
    def _cheap_cls_attention(self, x, blk):
        B, N, D = x.shape
        attn = blk.attn

        qkv = attn.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        q_cls = q[:, :, :1]  # CLS only
        attn_cls = (q_cls @ k.transpose(-2, -1)) * attn.scale
        attn_cls = attn_cls.softmax(dim=-1)

        return attn_cls, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            m = self.base_model
            B = x.size(0)
            device = x.device

            # ---- Patch embed ----
            x = m.patch_embed(x)

            # ---- CLS + Pos (SAFE) ----
            cls = m.cls_token.expand(B, -1, -1).to(device)
            x = torch.cat([cls, x], dim=1)
            x = x + m.pos_embed[:, : x.size(1)].to(device)
            x = m.pos_drop(x)

            N = x.size(1) - 1
            token_counts: List[int] = []

            prev_mass = torch.tensor(1.0, device=device)

            for i, blk in enumerate(m.blocks):
                token_counts.append(x.size(1))

                # ---------- cheap prune ----------
                if N > self.min_tokens:
                    x_norm = blk.norm1(x)
                    attn_cls, v = self._cheap_cls_attention(x_norm, blk)

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
                        keep_idx = select_tokens(importance, N_next, device)
                        x = torch.index_select(x, 1, keep_idx)
                        N = N_next

                    prev_mass = mass

                # ---------- full block ----------
                x = x + blk.drop_path1(blk.attn(blk.norm1(x)))
                x = x + blk.drop_path2(blk.mlp(blk.norm2(x)))

            x = m.norm(x)
            logits = m.head(x[:, 0])

            if self.collect_stats:
                self._last_stats = {"token_counts": token_counts}

            return logits