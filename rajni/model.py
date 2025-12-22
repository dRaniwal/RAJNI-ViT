import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class AdaptiveJacobianPrunedViT(nn.Module):
    """
    RAJNI: Adaptive Jacobian-based Token Pruning for Vision Transformers.

    Key ideas implemented here:
    - Patch importance from CLS Jacobian approximation
    - V-centering to remove positional bias
    - Adaptive pruning budget using derived CLS sensitivity ratio (rho)
    - No training required, inference-only modification

    This wrapper assumes a standard timm ViT / DeiT-style model.
    """

    def __init__(
        self,
        model,
        gamma=0.01,
        min_tokens=16,
        eps=1e-6,
    ):
        super().__init__()

        self.m = model
        self.blocks = model.blocks

        self.gamma = gamma
        self.min_tokens = min_tokens
        self.eps = eps

        # Automatically inferred properties
        self.embed_dim = model.embed_dim
        self.num_heads = model.blocks[0].attn.num_heads

        # For analysis & visualization
        self.last_kept_indices = []
        self.last_token_counts = []
        self.last_flops = []

    def forward(self, x):
        B = x.size(0)

        # ---- Patch embedding + positional encoding (unchanged) ----
        with autocast(device_type="cuda"):
            x = self.m.patch_embed(x)
            x = self.m._pos_embed(x)
            x = self.m.patch_drop(x)

        # Number of patch tokens (excluding CLS)
        N = x.size(1) - 1

        # Reset logging buffers every forward
        self.last_kept_indices = []
        self.last_token_counts = []
        self.last_flops = []

        prev_mass = None

        for layer_id, blk in enumerate(self.blocks):

            # =========================
            # Attention forward pass
            # =========================
            with autocast(device_type="cuda"):
                x_norm = blk.norm1(x)
                attn = blk.attn

                qkv = attn.qkv(x_norm).reshape(
                    B, x.size(1), 3, self.num_heads, -1
                ).permute(2, 0, 3, 1, 4)

                q, k, v = qkv[0], qkv[1], qkv[2]

                # Full attention (used for actual output)
                A = (q @ k.transpose(-2, -1)) * attn.scale
                A = A.softmax(dim=-1)

                out = (A @ v).transpose(1, 2).reshape(B, x.size(1), -1)
                out = attn.proj(out)

                x = x + blk.drop_path1(out)
                x = x + blk.drop_path2(blk.mlp(blk.norm2(x)))

            # Record token count (including CLS)
            self.last_token_counts.append(N + 1)

            # Stop pruning if token budget is already small
            if N <= self.min_tokens:
                continue

            # =========================
            # === Derived rho (CLS sensitivity)
            # =========================
            A_mean = A.mean(1)                 # [B, N+1, N+1]
            A_cls_cls = A_mean[:, 0, 0]        # self-attention of CLS

            V_cls = v.mean(1)[:, 0]            # CLS value vector
            V_cls_norm = V_cls.norm(dim=-1)

            # Derived sensitivity ratio:
            # rho_l ≈ 1 + A_cls->cls * ||V_cls||
            rho = (1.0 + A_cls_cls * V_cls_norm).mean()

            # =========================
            # Patch importance (Jacobian-based)
            # =========================
            A_cls = A_mean[:, 0, 1:N+1]         # CLS → patch attention
            V_pt = v.mean(1)[:, 1:N+1]          # patch value vectors

            # V-centering to remove positional dominance
            V_centered = V_pt - V_pt.mean(dim=1, keepdim=True)
            V_norm = V_centered.norm(dim=-1)

            mu = V_norm.mean(dim=1, keepdim=True)
            std = V_norm.std(dim=1, keepdim=True)

            V_gate = F.relu((V_norm - mu) / (std + self.eps))
            J = A_cls * V_gate                  # final patch importance

            mass = J.sum(dim=1).mean()

            # =========================
            # Adaptive pruning budget
            # =========================
            if prev_mass is not None:
                eta = mass / (prev_mass + self.eps)
                keep_ratio = (rho * eta).clamp(0.25, 4.0) ** (-self.gamma)

                N_next = int(N * keep_ratio)
                N_next = max(self.min_tokens, min(N_next, N))
            else:
                N_next = N  # first layer: no pruning

            # =========================
            # Token selection
            # =========================
            if N_next < N:
                score = J.mean(0)
                _, idx = torch.topk(score, k=N_next)

                idx = idx.sort().values + 1  # shift for CLS token
                keep_idx = torch.cat([
                    torch.zeros(1, device=x.device, dtype=torch.long),
                    idx
                ])

                self.last_kept_indices.append(keep_idx.detach().cpu())
                x = x[:, keep_idx]
                N = N_next

            prev_mass = mass

        # ---- Final normalization + head ----
        with autocast(device_type="cuda"):
            x = self.m.norm(x)
            logits = self.m.head(x[:, 0])

        return logits