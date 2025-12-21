#!/usr/bin/env python3
"""
Benchmarking script for RAJNI-ViT models.

This script benchmarks different pruning configurations and compares
throughput, latency, and accuracy trade-offs.
"""

import argparse
import sys
from pathlib import Path
import json

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rajni import RAJNIViT, setup_logger, load_config
from rajni.utils.metrics import calculate_throughput, calculate_flops


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark RAJNI-ViT configurations")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to base configuration YAML file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./benchmark_results.json",
        help="Path to save benchmark results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for throughput measurement"
    )
    
    return parser.parse_args()


def create_dummy_vit(num_classes: int = 1000):
    """
    Create a dummy ViT model for benchmarking.
    
    In practice, use timm to load actual models.
    """
    class DummyViT(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.embed_dim = 768
            self.num_classes = num_classes
            self.fc = nn.Linear(self.embed_dim, num_classes)
        
        def forward(self, x):
            batch_size = x.shape[0]
            features = torch.randn(batch_size, self.embed_dim, device=x.device)
            return self.fc(features)
    
    return DummyViT(num_classes)


def benchmark_configuration(
    base_vit,
    pruning_ratio,
    num_pruning_layers,
    importance_metric,
    device,
    batch_size,
    image_size,
    num_iterations,
    logger
):
    """
    Benchmark a specific RAJNI configuration.
    
    Args:
        base_vit: Base ViT model
        pruning_ratio: Token pruning ratio
        num_pruning_layers: Number of layers to prune
        importance_metric: Importance metric to use
        device: Device to run on
        batch_size: Batch size for benchmarking
        image_size: Input image size
        num_iterations: Number of iterations
        logger: Logger instance
        
    Returns:
        results: Dictionary with benchmark results
    """
    logger.info(f"\nBenchmarking configuration:")
    logger.info(f"  Pruning ratio: {pruning_ratio}")
    logger.info(f"  Pruning layers: {num_pruning_layers}")
    logger.info(f"  Importance metric: {importance_metric}")
    
    # Create RAJNI wrapper
    model = RAJNIViT(
        vit_model=base_vit,
        pruning_ratio=pruning_ratio,
        num_pruning_layers=num_pruning_layers,
        keep_cls_token=True,
        importance_metric=importance_metric
    )
    model = model.to(device)
    model.eval()
    
    # Create sample input
    sample_input = torch.randn(batch_size, 3, image_size, image_size).to(device)
    
    # Measure throughput
    throughput_metrics = calculate_throughput(
        model,
        sample_input,
        num_iterations=num_iterations,
        warmup_iterations=10
    )
    
    # Estimate FLOPs
    flops_metrics = calculate_flops(
        model,
        input_shape=(batch_size, 3, image_size, image_size),
        device=str(device)
    )
    
    results = {
        "pruning_ratio": pruning_ratio,
        "num_pruning_layers": num_pruning_layers,
        "importance_metric": importance_metric,
        "throughput_imgs_per_sec": throughput_metrics["throughput_imgs_per_sec"],
        "latency_ms": throughput_metrics["latency_sec"] * 1000,
        "total_params": flops_metrics["total_params"],
        "estimated_flops": flops_metrics["estimated_flops"],
    }
    
    logger.info(f"Results:")
    logger.info(f"  Throughput: {results['throughput_imgs_per_sec']:.2f} imgs/sec")
    logger.info(f"  Latency: {results['latency_ms']:.2f} ms")
    logger.info(f"  Total params: {results['total_params']:,}")
    
    return results


def main():
    """Main benchmarking function."""
    args = parse_args()
    
    # Load base configuration
    config = load_config(args.config)
    
    # Setup logger
    log_file = Path(config.logging.log_dir) / "benchmark.log"
    logger = setup_logger(
        name="rajni_benchmark",
        log_file=str(log_file),
        log_level=getattr(__import__('logging'), config.logging.log_level)
    )
    
    logger.info("=" * 80)
    logger.info("RAJNI-ViT Benchmarking Suite")
    logger.info("=" * 80)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(config.seed)
    
    # Create base ViT model
    logger.info(f"Loading base model: {config.model.name}")
    base_vit = create_dummy_vit(config.model.num_classes)
    
    # Define configurations to benchmark
    benchmark_configs = [
        {"pruning_ratio": 0.0, "num_pruning_layers": 0, "importance_metric": "norm"},
        {"pruning_ratio": 0.15, "num_pruning_layers": 4, "importance_metric": "jacobian"},
        {"pruning_ratio": 0.3, "num_pruning_layers": 6, "importance_metric": "jacobian"},
        {"pruning_ratio": 0.5, "num_pruning_layers": 8, "importance_metric": "jacobian"},
        {"pruning_ratio": 0.3, "num_pruning_layers": 6, "importance_metric": "attention"},
        {"pruning_ratio": 0.3, "num_pruning_layers": 6, "importance_metric": "norm"},
    ]
    
    all_results = []
    
    # Benchmark each configuration
    for cfg in benchmark_configs:
        results = benchmark_configuration(
            base_vit=base_vit,
            pruning_ratio=cfg["pruning_ratio"],
            num_pruning_layers=cfg["num_pruning_layers"],
            importance_metric=cfg["importance_metric"],
            device=device,
            batch_size=config.eval.batch_size,
            image_size=config.model.image_size,
            num_iterations=args.iterations,
            logger=logger
        )
        all_results.append(results)
    
    # Save results to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nBenchmark results saved to: {output_path}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Benchmark Summary")
    logger.info("=" * 80)
    logger.info(f"{'Pruning Ratio':<15} {'Layers':<10} {'Metric':<12} {'Throughput':<15} {'Latency':<12}")
    logger.info("-" * 80)
    
    for result in all_results:
        logger.info(
            f"{result['pruning_ratio']:<15.2f} "
            f"{result['num_pruning_layers']:<10} "
            f"{result['importance_metric']:<12} "
            f"{result['throughput_imgs_per_sec']:<15.2f} "
            f"{result['latency_ms']:<12.2f}"
        )
    
    logger.info("=" * 80)
    logger.info("Benchmarking completed!")


if __name__ == "__main__":
    main()
