import torch
import torch.nn as nn
from .attention import RAJNIAttention

class RAJNIViTWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, pruning_schedule: Dict[int, Dict]):
        super().__init__()
        self.m = base_model
        self.blocks = base_model.blocks
        self.pruning_schedule = pruning_schedule

        for i, blk in enumerate(self.blocks):
            if i in pruning_schedule:
                cfg = pruning_schedule[i]
                blk.attn = RAJNIAttention(
                    blk.attn,
                    keep_ratio=cfg["keep_ratio"],
                    update=cfg.get("update", True),
                )
                blk.has_pruner = True
            else:
                blk.has_pruner = False

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

        scores = None
        token_counts = []

        for i, blk in enumerate(self.blocks):
            token_counts.append(x.size(1))
            
            dp1 = blk.drop_path1 if hasattr(blk, "drop_path1") else nn.Identity()
            dp2 = blk.drop_path2 if hasattr(blk, "drop_path2") else nn.Identity()
            ls1 = blk.ls1 if hasattr(blk, "ls1") else nn.Identity()
            ls2 = blk.ls2 if hasattr(blk, "ls2") else nn.Identity()

            if blk.has_pruner:
                x_norm = blk.norm1(x)

                out, keep_idx, scores = blk.attn(x_norm, scores)

                gather_x = keep_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                x = torch.gather(x, 1, gather_x)

                x = x + dp1(ls1(out))
                x = x + dp2(ls2(blk.mlp(blk.norm2(x))))

            else:
                x = blk(x)
                scores = None  # invalidate importance if layout unchanged

        x = self.m.norm(x)
        x = self.m.head(x[:, 0])

        self._last_stats = {"token_counts": token_counts}
        return x