"""
Benchmarking utilities for RAJNI models.

Provides accuracy and throughput evaluation for pruned models.

Note on timing: We use torch.cuda.Event for accurate GPU timing,
which avoids host-device synchronization overhead that affects time.time().
"""
import time
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def _unwrap_model(m):
    """
    Unwrap model from DataParallel and/or torch.compile wrappers.
    
    Handles any combination in any order:
    - compile(DataParallel(model))
    - DataParallel(compile(model))  
    - compile(model)
    - DataParallel(model)
    - model (no wrapper)
    """
    while True:
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        elif isinstance(m, torch.nn.DataParallel):
            m = m.module
        else:
            break
    return m


def _warmup_model(model: nn.Module, dataloader: DataLoader, device: str, num_batches: int = 5):
    """
    Warmup model to ensure CUDA kernels are compiled and cached.
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
    
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    warmup: int = 5,
    max_batches: Optional[int] = None,
    use_cuda_events: bool = True,
) -> Tuple[float, float, Optional[Dict[str, Any]]]:
    """
    Benchmark a RAJNI-wrapped model for accuracy and throughput.
    
    Args:
        model: RAJNI-wrapped Vision Transformer (can be wrapped in DataParallel/compile)
        dataloader: Validation data loader
        device: Target device ("cuda" or "cpu")
        warmup: Number of warmup batches (not timed)
        max_batches: Maximum batches to process (None = all)
        use_cuda_events: Use CUDA events for timing (more accurate)
    
    Returns:
        accuracy: Top-1 accuracy as percentage
        throughput: Images per second
        stats: Pruning statistics from last batch
    """
    # Get the underlying model for stats access
    base_model = _unwrap_model(model)
    
    # Handle device placement
    unwrapped_check = model
    if hasattr(unwrapped_check, '_orig_mod'):
        unwrapped_check = unwrapped_check._orig_mod
    if not isinstance(unwrapped_check, torch.nn.DataParallel):
        model.to(device)
    model.eval()
    
    # Dedicated warmup phase
    print(f"  Warming up ({warmup} batches)...")
    _warmup_model(model, dataloader, device, warmup)
    
    correct = 0
    total = 0
    total_images = 0
    total_time = 0.0
    last_stats = None
    
    # CUDA events for accurate timing
    if use_cuda_events and device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    
    pbar = tqdm(dataloader, desc="RAJNI Benchmark", leave=False)
    
    for i, (images, labels) in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Timing with CUDA events
        if use_cuda_events and device == "cuda":
            start_event.record()
        else:
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
        
        logits = model(images)
        
        # Retrieve stats from base model (unwrapped once at start)
        last_stats = base_model.get_last_stats()
        
        # End timing
        if use_cuda_events and device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            total_time += elapsed_ms / 1000.0
        else:
            if device == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - start
        
        total_images += images.size(0)
        
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        if total > 0:
            pbar.set_postfix(acc=f"{100 * correct / total:.2f}%")
    
    accuracy = (correct / max(total, 1)) * 100.0
    throughput = total_images / max(total_time, 1e-6)
    
    return accuracy, throughput, last_stats
