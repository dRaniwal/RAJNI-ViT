import torch
import math

@torch.no_grad()
def compute_importance(qkv, num_heads,eps=1e-6):
    """
    qkv: [B, N, 3*C]
    returns: importance [B, N]
    """
    B, N, threeC = qkv.shape
    C = threeC // 3
    D = C // num_heads

    qkv = qkv.reshape(B, N, 3, num_heads, D).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]          # [B, H, N, D]

    # ---- CLS attention ----
    q_cls = q[:, :, 0:1, :]                   # [B, H, 1, D]
    logits = (q_cls @ k.transpose(-2, -1)) / math.sqrt(D)
    attn = logits.softmax(dim=-1)             # [B, H, 1, N]
    A_cls = attn.mean(dim=1).squeeze(1)       # [B, N]

    # ---- Value magnitude signal ----
    V = v.mean(dim=1)                         # [B, N, D]
    V = V - V.mean(dim=1, keepdim=True)       # center across tokens

    V_norm = V.norm(dim=-1)                   # [B, N]
    mu = V_norm.mean(dim=1, keepdim=True)
    std = V_norm.std(dim=1, keepdim=True) + eps

    z = (V_norm - mu) / std
    z = torch.sigmoid(z)                      # [B, N]

    return A_cls * z
