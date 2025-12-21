"""
Metrics and benchmarking utilities for RAJNI-ViT.
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Any, Optional


def calculate_flops(
    model: nn.Module,
    input_shape: tuple,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Estimate FLOPs for a model (placeholder implementation).
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (B, C, H, W)
        device: Device to run on
        
    Returns:
        flops_dict: Dictionary with FLOPs information
    """
    # Placeholder: In practice, use tools like fvcore or ptflops
    # This is a simplified estimation
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "total_params": total_params,
        "estimated_flops": total_params * 2,  # Rough estimate
        "input_shape": input_shape
    }


def calculate_throughput(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_iterations: int = 100,
    warmup_iterations: int = 10
) -> Dict[str, float]:
    """
    Measure model throughput (images/second).
    
    Args:
        model: PyTorch model
        input_tensor: Sample input tensor
        num_iterations: Number of iterations for measurement
        warmup_iterations: Number of warmup iterations
        
    Returns:
        throughput_dict: Dictionary with throughput metrics
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(input_tensor)
    
    # Synchronize for accurate timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_tensor)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    batch_size = input_tensor.shape[0]
    throughput = (num_iterations * batch_size) / elapsed_time
    latency = elapsed_time / num_iterations
    
    return {
        "throughput_imgs_per_sec": throughput,
        "latency_sec": latency,
        "batch_size": batch_size,
        "num_iterations": num_iterations
    }


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    topk: tuple = (1, 5)
) -> Dict[str, float]:
    """
    Compute top-k accuracy.
    
    Args:
        predictions: Model predictions [B, num_classes]
        targets: Ground truth labels [B]
        topk: Tuple of k values for top-k accuracy
        
    Returns:
        accuracy_dict: Dictionary with accuracy metrics
    """
    maxk = max(topk)
    batch_size = targets.size(0)
    
    _, pred = predictions.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    
    result = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        accuracy = correct_k.mul_(100.0 / batch_size).item()
        result[f"top{k}_accuracy"] = accuracy
    
    return result
