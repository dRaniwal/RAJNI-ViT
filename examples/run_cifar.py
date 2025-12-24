"""
RAJNI CIFAR-10/100 Evaluation Script.

Quick evaluation on CIFAR datasets for development and testing.
Uses a ViT model fine-tuned on CIFAR or applies a pretrained
ImageNet model with adjusted input size.

Usage:
    python run_cifar.py --dataset cifar10 --model vit_small_patch16_224
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from rajni import AdaptiveJacobianPrunedViT
from evaluation import benchmark, baseline_benchmark, flops_reduction


def get_args():
    parser = argparse.ArgumentParser(
        description="RAJNI CIFAR Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Path to download/load CIFAR",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit_small_patch16_224",
        help="timm model name",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.01,
        help="Pruning aggressiveness",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=16,
        help="Minimum tokens to keep",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Maximum batches (None = all)",
    )
    
    return parser.parse_args()


def create_dataloader(args):
    """Create CIFAR test dataloader with ViT-compatible transforms."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    if args.dataset == "cifar10":
        dataset = datasets.CIFAR10(
            args.data_path,
            train=False,
            download=True,
            transform=transform,
        )
        num_classes = 10
    else:
        dataset = datasets.CIFAR100(
            args.data_path,
            train=False,
            download=True,
            transform=transform,
        )
        num_classes = 100
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    return loader, num_classes


def main():
    args = get_args()
    
    print(f"RAJNI {args.dataset.upper()} Evaluation")
    print(f"Model: {args.model}, Gamma: {args.gamma}")
    print("-" * 40)
    
    # Create dataloader
    val_loader, num_classes = create_dataloader(args)
    print(f"Test samples: {len(val_loader.dataset)}")
    
    # Create model (note: using ImageNet pretrained, not CIFAR-tuned)
    base_model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=num_classes,
    )
    base_model.to(args.device)
    
    # Baseline
    print("\nBaseline evaluation...")
    acc_base, speed_base = baseline_benchmark(
        base_model,
        val_loader,
        device=args.device,
        max_batches=args.max_batches,
    )
    print(f"Baseline: {100 * acc_base:.2f}% @ {speed_base:.1f} img/s")
    
    # RAJNI
    print("\nRAJNI evaluation...")
    rajni_model = AdaptiveJacobianPrunedViT(
        base_model,
        gamma=args.gamma,
        min_tokens=args.min_tokens,
    )
    rajni_model.to(args.device)
    
    acc_rajni, speed_rajni, stats = benchmark(
        rajni_model,
        val_loader,
        device=args.device,
        max_batches=args.max_batches,
    )
    
    flops_stats = flops_reduction(rajni_model.m, stats)
    
    print(f"RAJNI: {100 * acc_rajni:.2f}% @ {speed_rajni:.1f} img/s")
    print(f"FLOPs reduction: {flops_stats['reduction_percent']:.1f}%")
    print(f"Speedup: {speed_rajni / speed_base:.2f}x")


if __name__ == "__main__":
    main()
