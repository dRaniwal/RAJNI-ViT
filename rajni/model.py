import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from .pruning import compute_jacobian_importance, select_tokens


class AdaptiveJacobianPrunedViT(nn.Module):
    """
    Fixed-schedule token pruning with TRUE attention cost reduction.

    Steps per layer:
    1. Compute cheap QKV
    2. Compute importance
    3. Prune tokens
    4. Compute FULL attention on reduced tokens
    """

    def __init__(
        self,
        model: nn.Module,
        r_max: float = 0.6,
        alpha: float = 1.5,
        min_tokens: int = 16,
        eps: float = 1e-6,
        collect_stats: bool = True,
    ):
        super().__init__()
        self.base_model = model
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
        frac = (layer_idx + 1) / self.num_layers
        keep_ratio = 1.0 - self.r_max * (frac ** self.alpha)
        return max(self.min_tokens, int(N0 * keep_ratio))

    # --------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.base_model
        B = x.size(0)

        # ---- Patch embed ----
        x = m.patch_embed(x)
        x = m._pos_embed(x)
        x = m.patch_drop(x)

        N0 = x.size(1) - 1
        N = N0

        token_counts: List[int] = []

        for layer_idx, blk in enumerate(m.blocks):
            token_counts.append(x.size(1))

            # ===============================
            # 1️⃣ CHEAP QKV FOR IMPORTANCE
            # ===============================
            x_norm = blk.norm1(x)
            attn = blk.attn

            qkv = attn.qkv(x_norm)
            qkv = qkv.reshape(
                B, x.size(1), 3, attn.num_heads, -1
            ).permute(2, 0, 3, 1, 4)

            q, k, v = qkv[0], qkv[1], qkv[2]

            attn_logits = (q @ k.transpose(-2, -1)) * attn.scale
            attn_probs = attn_logits.softmax(dim=-1)

            # ===============================
            # 2️⃣ IMPORTANCE
            # ===============================
            target_N = self._target_tokens(layer_idx, N0)

            if N > target_N:
                importance, _ = compute_jacobian_importance(
                    attn_probs, v, N, eps=self.eps
                )

                keep_idx = select_tokens(
                    importance, target_N, device=x.device
                )

                x = x[:, keep_idx]
                q = q[:, :, keep_idx]
                k = k[:, :, keep_idx]
                v = v[:, :, keep_idx]

                N = target_N

            # ===============================
            # 3️⃣ FULL ATTENTION (REDUCED)
            # ===============================
            attn_logits = (q @ k.transpose(-2, -1)) * attn.scale
            attn_probs = attn_logits.softmax(dim=-1)
            attn_out = attn_probs @ v

            attn_out = (
                attn_out.transpose(1, 2)
                .reshape(B, x.size(1), -1)
            )
            attn_out = attn.proj(attn_out)
            attn_out = attn.proj_drop(attn_out)

            x = x + blk.drop_path1(attn_out)

            # ===============================
            # 4️⃣ MLP
            # ===============================
            x = x + blk.drop_path2(blk.mlp(blk.norm2(x)))

        # ---- Head ----
        x = m.norm(x)
        logits = m.head(x[:, 0])

        self._last_stats = {"token_counts": token_counts}
        return logits