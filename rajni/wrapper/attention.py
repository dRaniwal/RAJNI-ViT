import torch
import torch.nn as nn
import math
from .importance import compute_importance


class RAJNIAttention(nn.Module):
    """
    RAJNI Attention with dynamic q-norm exponential scheduling.
    No fixed keep_ratio — computed per-batch based on layer difficulty.
    """
    def __init__(
        self,
        attn: nn.Module,
        layer_idx: int,
        *,
        percentile=0.75,
        kr_min=0.60,
        gamma=2.5,
        skip_layers=(10, 11),
    ):
        super().__init__()
        self.num_heads = attn.num_heads
        self.scale = attn.scale
        self.qkv = attn.qkv
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

        self.layer_idx = layer_idx
        self.percentile = percentile
        self.kr_min = kr_min
        self.gamma = gamma
        self.skip_layers = skip_layers

    def forward(self, x):
        """
        x: [B, N, C]
        Returns: (out, keep_idx)
        """
        B, N, C = x.shape

        # ==================================================
        # 🔒 Skip layers → full attention (no pruning)
        # ==================================================
        if self.layer_idx in self.skip_layers:
            qkv = self.qkv(x)
            qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)

            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            out = self.proj(out)
            out = self.proj_drop(out)

            # Return all indices (no pruning)
            keep_idx = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)
            return out, keep_idx

        # ==================================================
        # 🧠 Dynamic scheduling: importance → q-norm → prune
        # ==================================================
        qkv = self.qkv(x)
        scores = compute_importance(qkv, self.num_heads)

        # ---- Compute q-norm layer difficulty ----
        patch_scores = scores[:, 1:] + 1e-12
        log_scores = patch_scores.log()

        q = torch.quantile(log_scores, self.percentile, dim=1, keepdim=True)
        diff = torch.clamp(q - log_scores, min=0.0)

        D_l = (diff.mean(dim=1) / q.abs().squeeze(1)).mean().item()

        # ---- Exponential keep ratio (dynamic) ----
        keep_ratio = max(self.kr_min, math.exp(-self.gamma * D_l))

        num_patches = N - 1
        keep = max(1, int(keep_ratio * num_patches))

        # ---- Token selection ----
        _, idx = torch.topk(patch_scores, keep, dim=1)
        idx = idx.sort(dim=1).values

        cls_idx = torch.zeros((B, 1), device=x.device, dtype=torch.long)
        keep_idx = torch.cat([cls_idx, idx + 1], dim=1)

        # ---- Prune QKV ----
        gather = keep_idx.unsqueeze(-1).expand(-1, -1, qkv.shape[-1])
        qkv = torch.gather(qkv, 1, gather)

        Np = qkv.shape[1]
        qkv = qkv.reshape(B, Np, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, Np, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out, keep_idx
