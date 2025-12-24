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


def _warmup_model(model: nn.Module, dataloader: DataLoader, device: str, num_batches: int = 10):
    """
    Warmup model to ensure CUDA kernels are compiled and cached.
    
    This is critical for fair benchmarking - the first few forward passes
    are always slower due to JIT compilation and memory allocation.
    """
    model.eval()
    data_iter = iter(dataloader)
    
    with torch.no_grad():
        for i in range(num_batches):
            try:
                images, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                images, _ = next(data_iter)
            
            images = images.to(device, non_blocking=True)
            _ = model(images)
    
    # Ensure all warmup operations complete
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def baseline_benchmark(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    warmup: int = 10,
    max_batches: Optional[int] = None,
    use_cuda_events: bool = True,
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
        use_cuda_events: Use CUDA events for timing (more accurate)
    
    Returns:
        accuracy: Top-1 accuracy as percentage
        throughput: Images per second
    """
    # Handle DataParallel wrapper
    if not isinstance(model, torch.nn.DataParallel):
        model.to(device)
    model.eval()
    
    # Dedicated warmup phase
    print(f"  Warming up ({warmup} batches)...")
    _warmup_model(model, dataloader, device, warmup)
    
    correct = 0
    total = 0
    total_images = 0
    total_time = 0.0
    
    # CUDA events for accurate timing
    if use_cuda_events and device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    
    pbar = tqdm(dataloader, desc="Baseline Benchmark", leave=False)
    
    for i, (images, labels) in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Timing with CUDA events (more accurate than time.time)
        if use_cuda_events and device == "cuda":
            start_event.record()
        else:
            torch.cuda.synchronize() if device == "cuda" else None
            start = time.time()
        
        logits = model(images)
        
        # End timing
        if use_cuda_events and device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            total_time += elapsed_ms / 1000.0
        else:
            torch.cuda.synchronize() if device == "cuda" else None
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
