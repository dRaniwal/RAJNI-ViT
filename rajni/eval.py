import time
import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device="cuda",
    max_batches=None,
    warmup=5,
):
    model.eval()
    model.to(device)

    # ---- Warmup ----
    print(f"Warming up {warmup} batches")
    it = iter(dataloader)
    for _ in range(warmup):
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(dataloader)
            x, _ = next(it)
        model(x.to(device))

    if device == "cuda":
        torch.cuda.synchronize()

    correct = 0
    total = 0
    total_images = 0
    total_time = 0.0

    # ---- Progress bar ----
    pbar = tqdm(
        dataloader,
        desc="Evaluating",
        total=max_batches if max_batches is not None else len(dataloader),
        leave=False,
    )

    for i, (images, labels) in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()

        logits = model(images)

        if device == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - start

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_images += images.size(0)

        # ---- Update progress bar ----
        if total > 0:
            pbar.set_postfix(
                acc=f"{100.0 * correct / total:.2f}%",
                imgs_per_s=f"{total_images / max(total_time, 1e-6):.1f}",
            )

    acc = 100.0 * correct / max(total, 1)
    throughput = total_images / max(total_time, 1e-6)

    return acc, throughput