"""
RAJNI ImageNet Evaluation Script.

Evaluates RAJNI-wrapped Vision Transformers on ImageNet validation set.
Reports accuracy, throughput, and FLOPs reduction.

Usage:
    python run_imagenet.py --data_path /path/to/imagenet --model vit_base_patch16_224

For hyperparameter sweeps, see scripts/sweep_gamma.sh
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rajni import AdaptiveJacobianPrunedViT
from evaluation import benchmark, baseline_benchmark, flops_reduction


def get_args():
    parser = argparse.ArgumentParser(
        description="RAJNI ImageNet Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to ImageNet dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )
    
    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="vit_base_patch16_224",
        help="timm model name",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (uses pretrained if None)",
    )
    
    # RAJNI parameters
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.01,
        help="Pruning aggressiveness (higher = more pruning)",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=16,
        help="Minimum tokens to keep per layer",
    )
    
    # Evaluation
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Maximum batches to evaluate (None = all)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup batches (not timed)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also run baseline (unpruned) evaluation",
    )
    
    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Use DataParallel for multi-GPU",
    )
    
    return parser.parse_args()


def create_dataloader(args):
    """Create ImageNet validation dataloader."""
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    val_path = Path(args.data_path) / "val"
    if not val_path.exists():
        val_path = Path(args.data_path)
    
    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return val_loader


def main():
    args = get_args()
    
    print("=" * 60)
    print("RAJNI ImageNet Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Gamma: {args.gamma}")
    print(f"Min tokens: {args.min_tokens}")
    print("=" * 60)
    
    # Create model
    print("\nLoading model...")
    if args.checkpoint:
        base_model = timm.create_model(args.model, checkpoint_path=args.checkpoint)
    else:
        base_model = timm.create_model(args.model, pretrained=True)
    
    # Create dataloader
    print("Creating dataloader...")
    val_loader = create_dataloader(args)
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Baseline evaluation
    if args.baseline:
        print("\n" + "-" * 40)
        print("Baseline (unpruned) evaluation")
        print("-" * 40)
        
        baseline_model = base_model
        if args.multi_gpu:
            baseline_model = nn.DataParallel(baseline_model)
        baseline_model.to(args.device)
        
        acc_base, speed_base = baseline_benchmark(
            baseline_model,
            val_loader,
            device=args.device,
            warmup=args.warmup,
            max_batches=args.max_batches,
        )
        
        print(f"Baseline Accuracy: {100 * acc_base:.2f}%")
        print(f"Baseline Throughput: {speed_base:.1f} img/s")
    
    # RAJNI evaluation
    print("\n" + "-" * 40)
    print("RAJNI (pruned) evaluation")
    print("-" * 40)
    
    rajni_model = AdaptiveJacobianPrunedViT(
        base_model,
        gamma=args.gamma,
        min_tokens=args.min_tokens,
    )
    
    if args.multi_gpu:
        rajni_model = nn.DataParallel(rajni_model)
    rajni_model.to(args.device)
    
    acc_rajni, speed_rajni, stats = benchmark(
        rajni_model,
        val_loader,
        device=args.device,
        warmup=args.warmup,
        max_batches=args.max_batches,
    )
    
    # Compute FLOPs reduction
    if args.multi_gpu:
        core_model = rajni_model.module.m
    else:
        core_model = rajni_model.m
    
    flops_stats = flops_reduction(core_model, stats)
    
    # Print results
    print(f"\nRAJNI Accuracy: {100 * acc_rajni:.2f}%")
    print(f"RAJNI Throughput: {speed_rajni:.1f} img/s")
    print(f"\nFLOPs Analysis:")
    print(f"  Baseline: {flops_stats['baseline_GFLOPs']:.2f} GFLOPs")
    print(f"  RAJNI: {flops_stats['rajni_GFLOPs']:.2f} GFLOPs")
    print(f"  Reduction: {flops_stats['reduction_percent']:.1f}%")
    
    # Token counts per layer
    print(f"\nToken counts per layer:")
    for i, count in enumerate(stats["token_counts"]):
        print(f"  Layer {i:2d}: {count:3d} tokens")
    
    # Summary comparison
    if args.baseline:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Accuracy drop: {100 * (acc_base - acc_rajni):.2f}%")
        print(f"Speedup: {speed_rajni / speed_base:.2f}x")
        print(f"FLOPs reduction: {flops_stats['reduction_percent']:.1f}%")


if __name__ == "__main__":
    main()
