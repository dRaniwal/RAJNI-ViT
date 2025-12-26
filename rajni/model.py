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
    """
    RAJNI: Adaptive Jacobian-based Token Pruning for Vision Transformers

    Core principles (SAINT-consistent):
    ----------------------------------
    1. Importance is computed using CLS-only attention (cheap).
    2. Tokens are pruned BEFORE full attention.
    3. Full attention is executed only on surviving tokens.
    4. No fake attention, no manual pos-embed, no kernel-breaking ops.

    This preserves FlashAttention / SDPA paths.
    """

    def __init__(
        self,
        model: nn.Module,
        gamma: float = 0.5,
        min_tokens: int = 16,
        eps: float = 1e-6,
        collect_stats: bool = True,
    ):
        super().__init__()

        # register base model correctly (DataParallel safe)
        self.base_model = model

        self.gamma = gamma
        self.min_tokens = min_tokens
        self.eps = eps
        self.collect_stats = collect_stats

        self.embed_dim = model.embed_dim
        self.num_heads = model.blocks[0].attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads

        self._last_stats: Optional[Dict[str, Any]] = None

    # --------------------------------------------------
    # Public API for benchmark / visualizer
    # --------------------------------------------------
    def get_last_stats(self):
        return self._last_stats

    # --------------------------------------------------
    # CLS-only attention (CHEAP, no expansion)
    # --------------------------------------------------
    def _cls_attention_and_values(self, x_norm, attn_module):
        """
        Computes:
            - CLS → all attention (A_cls)
            - value vectors (v)

        Shapes:
            A_cls: [B, H, 1, N]
            v:     [B, H, N, D]
        """
        B, N, _ = x_norm.shape

        qkv = attn_module.qkv(x_norm)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        q_cls = q[:, :, :1]  # CLS only
        attn_cls = (q_cls @ k.transpose(-2, -1)) * attn_module.scale
        attn_cls = attn_cls.softmax(dim=-1)

        return attn_cls, v

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            m = self.base_model
            B = x.size(0)

            # ---- Patch + CLS + Pos (SAFE, timm-native) ----
            x = m.patch_embed(x)
            x = m._pos_embed(x)
            x = m.patch_drop(x)

            N = x.size(1) - 1  # patch tokens only

            token_counts: List[int] = []
            prev_mass = torch.tensor(1.0, device=x.device)

            for layer_idx, blk in enumerate(m.blocks):
                token_counts.append(x.size(1))

                # --------------------------------------------------
                # PRUNE BEFORE FULL ATTENTION
                # --------------------------------------------------
                if N > self.min_tokens:
                    x_norm = blk.norm1(x)

                    # cheap CLS-only signal
                    attn_cls, v = self._cls_attention_and_values(x_norm, blk.attn)

                    # CLS sensitivity (rho)
                    # rho = compute_cls_sensitivity(
                    #     attention=attn_cls,
                    #     values=v,
                    #     layer_idx=layer_idx,
                    # )

                    # Jacobian importance (patch-level)
                    importance, mass = compute_jacobian_importance(
                        attention=attn_cls,
                        values=v,
                        num_patches=N,
                        eps=self.eps,
                    )

                    # adaptive keep ratio
                    keep_ratio = compute_keep_ratio(
                        # rho=rho,
                        current_mass=mass,
                        prev_mass=prev_mass,
                        gamma=self.gamma,
                        eps=self.eps,
                    )

                    keep_ratio_scalar = float(keep_ratio.clamp(0.0, 1.0).item())
                    N_next = max(self.min_tokens, int(N * keep_ratio_scalar))

                    # actual pruning
                    if N_next < N:
                        keep_idx = select_tokens(
                            importance=importance,
                            num_keep=N_next,
                            device=x.device,
                        )
                        x = torch.index_select(x, dim=1, index=keep_idx)
                        N = N_next

                    prev_mass = mass

                # --------------------------------------------------
                # FULL BLOCK (FUSED ATTENTION PATH)
                # --------------------------------------------------
                x = x + blk.drop_path1(blk.attn(blk.norm1(x)))
                x = x + blk.drop_path2(blk.mlp(blk.norm2(x)))

            # ---- Head ----
            x = m.norm(x)
            logits = m.head(x[:, 0])

            if self.collect_stats:
                self._last_stats = {
                    "token_counts": token_counts,
                }
            else:
                self._last_stats = None

            return logits

    def __repr__(self):
        return (
            f"AdaptiveJacobianPrunedViT("
            f"gamma={self.gamma}, "
            f"min_tokens={self.min_tokens}, "
            f"layers={len(self.base_model.blocks)}, "
            f"embed_dim={self.embed_dim})"
        )