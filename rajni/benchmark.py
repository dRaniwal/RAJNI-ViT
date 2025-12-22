import time
import torch
from tqdm import tqdm

from .utils import unwrap_model
from .flops import flops_reduction


@torch.no_grad()
def benchmark(
    model,
    dataloader,
    device="cuda",
    warmup=5,
    max_batches=None,
):
    model.eval()
    device = torch.device(device)

    correct = 0
    total = 0
    total_images = 0
    total_time = 0.0

    for i, (x, y) in enumerate(tqdm(dataloader, desc="RAJNI Benchmark")):

        if max_batches and i >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if i >= warmup:
            torch.cuda.synchronize()
            start = time.time()

        logits = model(x)

        if i >= warmup:
            torch.cuda.synchronize()
            total_time += time.time() - start
            total_images += x.size(0)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    acc = correct / total
    speed = total_images / total_time

    core = unwrap_model(model)
    flops = flops_reduction(core)

    print("\n========== RAJNI RESULTS ==========")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Speed: {speed:.1f} img/s")
    print(f"Baseline FLOPs: {flops['baseline_GFLOPs']:.2f} GF")
    print(f"RAJNI FLOPs: {flops['rajni_GFLOPs']:.2f} GF")
    print(f"Reduction: {flops['reduction_%']:.2f}%")
    print("==================================")

    return acc, speed, flops