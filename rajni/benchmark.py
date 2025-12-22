import time
import torch
from tqdm import tqdm


@torch.no_grad()
def benchmark(model, dataloader, device="cuda", warmup=5, max_batches=None):
    """
    Benchmarks accuracy + throughput.
    Works with both DataParallel and single GPU.
    """

    model.eval()
    model.to(device)

    correct = 0
    total = 0
    total_images = 0
    total_time = 0.0

    last_stats = None

    for i, (x, y) in enumerate(tqdm(dataloader, desc="RAJNI Benchmark")):
        if max_batches and i >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if i >= warmup:
            torch.cuda.synchronize()
            start = time.time()

        logits = model(x)

        # Get stats from the model (DataParallel-safe)
        if isinstance(model, torch.nn.DataParallel):
            last_stats = model.module.get_last_stats()
        else:
            last_stats = model.get_last_stats()

        if i >= warmup:
            torch.cuda.synchronize()
            total_time += time.time() - start
            total_images += x.size(0)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    acc = correct / max(total, 1)
    speed = total_images / max(total_time, 1e-6)

    return acc, speed, last_stats