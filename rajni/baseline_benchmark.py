import time
import torch
from tqdm import tqdm


@torch.no_grad()
def baseline_benchmark(
    model,
    dataloader,
    device="cuda",
    warmup=5,
    max_batches=None,
):
    """
    Benchmark a vanilla ViT / DeiT model.

    Measures:
    - Top-1 accuracy
    - Throughput (images / second)

    IMPORTANT:
    - Uses the same timing logic as RAJNI benchmark
    - No pruning, no stats, no hooks
    """

    model.eval()
    model.to(device)

    correct = 0
    total = 0

    total_images = 0
    total_time = 0.0

    for i, (x, y) in enumerate(
        tqdm(dataloader, desc="Baseline Benchmark", leave=False)
    ):
        if max_batches is not None and i >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # -------------------------
        # Warmup (do not time)
        # -------------------------
        if i >= warmup:
            torch.cuda.synchronize()
            start = time.time()

        logits = model(x)

        if i >= warmup:
            torch.cuda.synchronize()
            total_time += time.time() - start
            total_images += x.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / max(total, 1)
    speed = total_images / max(total_time, 1e-6)

    return acc, speed