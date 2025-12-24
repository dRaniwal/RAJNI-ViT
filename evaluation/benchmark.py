"""
Benchmarking utilities for RAJNI models.

Provides accuracy and throughput evaluation for pruned models.
"""
import time
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def benchmark(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    warmup: int = 5,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, Optional[Dict[str, Any]]]:
    """
    Benchmark a RAJNI-wrapped model for accuracy and throughput.
    
    Measures top-1 accuracy and images per second, while collecting
    pruning statistics from the model.
    
    Args:
        model: RAJNI-wrapped Vision Transformer
        dataloader: Validation data loader
        device: Target device ("cuda" or "cpu")
        warmup: Number of warmup batches (not timed)
        max_batches: Maximum batches to process (None = all)
    
    Returns:
        accuracy: Top-1 accuracy (0.0 to 1.0)
        throughput: Images per second
        stats: Pruning statistics from last batch
    """
    # Handle DataParallel wrapper
    if not isinstance(model, torch.nn.DataParallel):
        model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    total_images = 0
    total_time = 0.0
    last_stats = None
    
    pbar = tqdm(dataloader, desc="RAJNI Benchmark", leave=False)
    
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
        
        # Retrieve stats (DataParallel-safe)
        if isinstance(model, torch.nn.DataParallel):
            last_stats = model.module.get_last_stats()
        else:
            last_stats = model.get_last_stats()
        
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
    
    return accuracy, throughput, last_stats
