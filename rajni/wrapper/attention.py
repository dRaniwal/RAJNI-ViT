import torch
import torch.nn as nn
from .importance import compute_importance

class RAJNIAttention(nn.Module):
    def __init__(self, attn: nn.Module, keep_ratio: float, update: bool):
        super().__init__()
        self.num_heads = attn.num_heads
        self.scale = attn.scale
        self.qkv = attn.qkv
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

        self.keep_ratio = keep_ratio
        self.update = update

    def forward(self, x, prev_scores=None):
        """
        x: [B, N, C]
        """
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, N, 3C]

        # ---- Importance ----
        if self.update or prev_scores is None:
            scores = compute_importance(qkv,self.num_heads)
        else:
            scores = prev_scores

        # ---- Token selection ----
        num_patches = N - 1
        keep = max(1, int(self.keep_ratio * num_patches))

        patch_scores = scores[:, 1:]
        _, idx = torch.topk(patch_scores, keep, dim=1)
        idx = torch.sort(idx, dim=1).values

        cls_idx = torch.zeros((B, 1), device=x.device, dtype=torch.long)
        keep_idx = torch.cat([cls_idx, idx + 1], dim=1)  # [B, K+1]

        # ---- Prune QKV ----
        gather_qkv = keep_idx.unsqueeze(-1).expand(-1, -1, qkv.shape[-1])
        qkv = torch.gather(qkv, 1, gather_qkv)

        # ---- Attention on reduced tokens ----
        Np = qkv.shape[1]
        qkv = qkv.reshape(B, Np, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, Np, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        next_scores = torch.gather(scores, 1, keep_idx)

        return out, keep_idx, next_scores
