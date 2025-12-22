import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class AdaptiveJacobianPrunedViT(nn.Module):
    """
    RAJNI: Adaptive Jacobian-based Token Pruning for Vision Transformers

    Design principles:
    - Inference-only (no training required)
    - DataParallel-safe (returns only tensors from forward)
    - Stats accessible via get_last_stats() method

    Forward returns:
        logits: Tensor [B, num_classes]
    
    Stats available via:
        model.get_last_stats() -> dict with token_counts & kept_indices
    """

    def __init__(self, model, gamma=0.01, min_tokens=16, eps=1e-6):
        super().__init__()
        self.m = model
        self.blocks = model.blocks

        self.gamma = gamma
        self.min_tokens = min_tokens
        self.eps = eps

        self.num_heads = self.blocks[0].attn.num_heads
        self._last_stats = None

    def get_last_stats(self):
        """Get stats from the last forward pass (DataParallel-safe)."""
        return self._last_stats

    def forward(self, x):
        B = x.size(0)

        # ---- Patch + CLS + Pos (unchanged ViT path) ----
        with autocast(device_type="cuda"):
            x = self.m.patch_embed(x)
            x = self.m._pos_embed(x)
            x = self.m.patch_drop(x)

        N = x.size(1) - 1  # patch tokens only

        token_counts = []
        kept_indices = []

        prev_mass = None

        for blk in self.blocks:

            # -------------------------
            # Attention forward
            # -------------------------
            with autocast(device_type="cuda"):
                x_norm = blk.norm1(x)
                attn = blk.attn

                qkv = attn.qkv(x_norm).reshape(
                    B, x.size(1), 3, self.num_heads, -1
                ).permute(2, 0, 3, 1, 4)

                q, k, v = qkv[0], qkv[1], qkv[2]

                A = (q @ k.transpose(-2, -1)) * attn.scale
                A = A.softmax(dim=-1)

                out = (A @ v).transpose(1, 2).reshape(B, x.size(1), -1)
                out = attn.proj(out)

                x = x + blk.drop_path1(out)
                x = x + blk.drop_path2(blk.mlp(blk.norm2(x)))

            # ---- Log token count ----
            token_counts.append(N + 1)  # include CLS

            if N <= self.min_tokens:
                continue

            # -------------------------
            # Derived CLS sensitivity (rho)
            # -------------------------
            A_mean = A.mean(1)
            A_cls_cls = A_mean[:, 0, 0]

            V_cls = v.mean(1)[:, 0]
            rho = (1.0 + A_cls_cls * V_cls.norm(dim=-1)).mean()

            # -------------------------
            # Jacobian patch importance
            # -------------------------
            A_cls = A_mean[:, 0, 1:N+1]
            V_pt = v.mean(1)[:, 1:N+1]

            # V-centering (critical)
            V_centered = V_pt - V_pt.mean(dim=1, keepdim=True)
            V_norm = V_centered.norm(dim=-1)

            mu = V_norm.mean(dim=1, keepdim=True)
            std = V_norm.std(dim=1, keepdim=True)

            J = A_cls * F.relu((V_norm - mu) / (std + self.eps))
            mass = J.sum(dim=1).mean()

            # -------------------------
            # Adaptive pruning budget
            # -------------------------
            if prev_mass is not None:
                eta = mass / (prev_mass + self.eps)
                keep_ratio = (rho * eta).clamp(0.25, 4.0) ** (-self.gamma)
                N_next = max(self.min_tokens, int(N * keep_ratio))
            else:
                N_next = N

            # -------------------------
            # Prune tokens
            # -------------------------
            if N_next < N:
                score = J.mean(0)
                _, idx = torch.topk(score, k=N_next)

                idx = idx.sort().values + 1  # shift for CLS
                keep_idx = torch.cat([
                    torch.zeros(1, device=x.device, dtype=torch.long),
                    idx
                ])

                kept_indices.append(keep_idx.detach().cpu())
                x = x[:, keep_idx]
                N = N_next

            prev_mass = mass

        # ---- Head ----
        with autocast(device_type="cuda"):
            x = self.m.norm(x)
            logits = self.m.head(x[:, 0])

        # Store stats for later retrieval (DataParallel-safe)
        self._last_stats = {
            "token_counts": token_counts,
            "kept_indices": kept_indices,
        }

        return logits