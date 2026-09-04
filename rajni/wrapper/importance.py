import torch
import math


@torch.inference_mode()
def compute_importance(qkv: torch.Tensor, num_heads: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute importance: CLS-attention × sigmoid-normalized |V|
    Fused to ~4 core operations for maximum GPU throughput.
    """
    B, N, C3 = qkv.shape
    C = C3 // 3
    D = C // num_heads
    scale = math.sqrt(D)  # Pre-compute once

    # Op 1: Split QKV (zero-copy views)
    q, k, v = qkv.split(C, dim=-1)

    # Op 2: CLS attention = softmax(q_cls @ k^T / sqrt(D)).mean(heads)
    A_cls = (
        q[:, :1]                                          # [B, 1, C]
        .view(B, 1, num_heads, D)                         # [B, 1, H, D]
        .transpose(1, 2)                                  # [B, H, 1, D]
        .matmul(k.view(B, N, num_heads, D).permute(0, 2, 3, 1))  # [B, H, 1, N]
        .div_(scale)
        .softmax(dim=-1)
        .mean(dim=1)                                      # [B, 1, N]
        .squeeze_(1)                                      # [B, N]
    )

    # Op 3: Value magnitude
    V = v.view(B, N, num_heads, D).mean(dim=2)            # [B, N, D]

    # L1 norm instead of L2 (no sqrt, no square)
    V_norm = (V - V.mean(dim=1, keepdim=True)).abs().mean(dim=-1)  # [B, N]

    # Op 4: Sigmoid normalization = sigmoid((x - μ) / σ)
    std, mu = torch.std_mean(V_norm, dim=1, keepdim=True, unbiased=False)
    return A_cls * torch.sigmoid((V_norm - mu) / (std + eps))
