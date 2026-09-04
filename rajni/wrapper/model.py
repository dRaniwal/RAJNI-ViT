import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from .attention import RAJNIAttention


class _BlockOps:
    """Cached block operations to avoid getattr overhead."""
    __slots__ = ('norm1', 'norm2', 'attn', 'mlp', 'dp1', 'dp2', 'ls1', 'ls2')

    def __init__(self, blk: nn.Module):
        self.norm1 = blk.norm1
        self.norm2 = blk.norm2
        self.attn = blk.attn
        self.mlp = blk.mlp
        self.dp1 = getattr(blk, 'drop_path1', None) or nn.Identity()
        self.dp2 = getattr(blk, 'drop_path2', None) or nn.Identity()
        self.ls1 = getattr(blk, 'ls1', None) or nn.Identity()
        self.ls2 = getattr(blk, 'ls2', None) or nn.Identity()


class RAJNIViTWrapper(nn.Module):
    """
    Optimized RAJNI ViT Wrapper with dynamic token pruning.
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        percentile: float = 0.75,
        kr_min: float = 0.60,
        gamma: float = 2.5,
        tau_warmup: float = 0.10,  # Passed down to dynamic scheduler
        compile_blocks: bool = False,
    ):
        super().__init__()
        self.m = base_model
        self.blocks = base_model.blocks
        self.num_blocks = len(self.blocks)

        # Replace attention layers
        for i, blk in enumerate(self.blocks):
            blk.attn = RAJNIAttention(
                blk.attn,
                layer_idx=i,
                percentile=percentile,
                kr_min=kr_min,
                gamma=gamma,
                tau_warmup=tau_warmup,  # Replaces static skip_layers
            )

            # Optional: compile MLP for extra throughput
            if compile_blocks and hasattr(torch, 'compile'):
                blk.mlp = torch.compile(blk.mlp, mode='reduce-overhead')

        # Cache block ops (avoids getattr per forward)
        self._ops: Tuple[_BlockOps, ...] = tuple(_BlockOps(blk) for blk in self.blocks)
        self._last_stats: Optional[Dict] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        x = self.m.patch_embed(x)
        x = torch.cat(
            [self.m.cls_token.expand(B, -1, -1), x],
            dim=1
        )

        x = self.m.pos_drop(
            x + self.m.pos_embed[:, :x.size(1)]
        )

        token_counts = []
        current_indices = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(B, -1)
        self.keep_history = {}

        for layer_id, ops in enumerate(self._ops):
            token_counts.append(x.size(1))

            out, keep_idx = ops.attn(ops.norm1(x))

            original_keep = current_indices.gather(1, keep_idx)
            self.keep_history[layer_id] = original_keep.detach().cpu()
            current_indices = original_keep

            if keep_idx.size(1) < x.size(1):
                x = x.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))

            x = x + ops.dp1(ops.ls1(out))
            x = x + ops.dp2(ops.ls2(ops.mlp(ops.norm2(x))))

        x = self.m.norm(x)
        x = self.m.head(x[:, 0])

        self._last_stats = {
            "token_counts": token_counts
        }

        return x

    @property
    def last_stats(self) -> Optional[Dict]:
        return self._last_stats
