import torch
import torch.nn as nn
from .attention import RAJNIAttention


class RAJNIViTWrapper(nn.Module):
    """
    RAJNI ViT Wrapper with fully dynamic scheduling.
    No fixed pruning schedule — all layers use dynamic q-norm scheduling.
    """
    def __init__(
        self,
        base_model: nn.Module,
        *,
        percentile=0.75,
        kr_min=0.60,
        gamma=2.5,
        skip_layers=(10, 11),
    ):
        super().__init__()
        self.m = base_model
        self.blocks = base_model.blocks

        # Apply RAJNIAttention to ALL layers (dynamic scheduling)
        for i, blk in enumerate(self.blocks):
            blk.attn = RAJNIAttention(
                blk.attn,
                layer_idx=i,
                percentile=percentile,
                kr_min=kr_min,
                gamma=gamma,
                skip_layers=skip_layers,
            )

        self._last_stats = None

    def get_last_stats(self):
        return self._last_stats

    def forward(self, x):
        B = x.size(0)

        # ---- Patch + CLS + Pos ----
        x = self.m.patch_embed(x)
        cls = self.m.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.m.pos_drop(x + self.m.pos_embed[:, :x.size(1)])

        token_counts = []

        for blk in self.blocks:
            token_counts.append(x.size(1))

            dp1 = getattr(blk, "drop_path1", nn.Identity())
            dp2 = getattr(blk, "drop_path2", nn.Identity())
            ls1 = getattr(blk, "ls1", nn.Identity())
            ls2 = getattr(blk, "ls2", nn.Identity())

            x_norm = blk.norm1(x)
            out, keep_idx = blk.attn(x_norm)

            # Gather residual tokens
            x = torch.gather(
                x, 1, keep_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            )

            x = x + dp1(ls1(out))
            x = x + dp2(ls2(blk.mlp(blk.norm2(x))))

        x = self.m.norm(x)
        x = self.m.head(x[:, 0])

        self._last_stats = {"token_counts": token_counts}
        return x