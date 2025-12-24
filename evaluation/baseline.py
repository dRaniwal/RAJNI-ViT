"""
Baseline benchmarking for vanilla Vision Transformers.

Provides comparison metrics against unpruned models.
"""
import time
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def baseline_benchmark(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    warmup: int = 5,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Benchmark a vanilla ViT/DeiT model for comparison.
    
    Uses identical timing methodology to ensure fair comparison
    with RAJNI-pruned models.
    
    Args:
        model: Vanilla Vision Transformer (not wrapped)
        dataloader: Validation data loader
        device: Target device ("cuda" or "cpu")
        warmup: Number of warmup batches (not timed)
        max_batches: Maximum batches to process (None = all)
    
    Returns:
        accuracy: Top-1 accuracy (0.0 to 1.0)
        throughput: Images per second
    """
    # Handle DataParallel wrapper
    if not isinstance(model, torch.nn.DataParallel):
        model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    total_images = 0
    total_time = 0.0
    
    pbar = tqdm(dataloader, desc="Baseline Benchmark", leave=False)
    
    for i, (images, labels) in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Warmup phase (not timed)
        if i >= warmup:
            torch.cuda.synchronize()
            start = time.time()
        
        logits = model(images)
        
        if i >= warmup:
            torch.cuda.synchronize()
            total_time += time.time() - start
            total_images += images.size(0)
        
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        if total > 0:
            pbar.set_postfix(acc=f"{100 * correct / total:.2f}%")

    accuracy = (correct / max(total, 1)) * 100.0
    throughput = total_images / max(total_time, 1e-6)
    
    return accuracy, throughput
