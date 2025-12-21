#!/usr/bin/env python3
"""
Evaluation script for RAJNI-ViT models.

This script evaluates a Vision Transformer model with RAJNI pruning
on a specified dataset and reports accuracy and efficiency metrics.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rajni import RAJNIViT, setup_logger, load_config
from rajni.utils.metrics import compute_accuracy, calculate_throughput


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate RAJNI-ViT model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu), overrides config"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size, overrides config"
    )
    
    return parser.parse_args()


def create_dummy_dataloader(batch_size: int, image_size: int, num_batches: int = 10):
    """
    Create a dummy dataloader for demonstration.
    
    In practice, replace this with actual dataset loading (ImageNet, etc.)
    """
    class DummyDataset:
        def __init__(self, num_samples, image_size, num_classes):
            self.num_samples = num_samples
            self.image_size = image_size
            self.num_classes = num_classes
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            image = torch.randn(3, self.image_size, self.image_size)
            label = torch.randint(0, self.num_classes, (1,)).item()
            return image, label
    
    dataset = DummyDataset(
        num_samples=batch_size * num_batches,
        image_size=image_size,
        num_classes=1000
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return dataloader


def create_dummy_vit(num_classes: int = 1000):
    """
    Create a dummy ViT model for demonstration.
    
    In practice, use timm or torchvision to load pre-trained models:
    import timm
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    """
    class DummyViT(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.embed_dim = 768
            self.num_classes = num_classes
            self.fc = nn.Linear(self.embed_dim, num_classes)
        
        def forward(self, x):
            # Dummy forward: average pool and classify
            batch_size = x.shape[0]
            features = torch.randn(batch_size, self.embed_dim, device=x.device)
            return self.fc(features)
    
    return DummyViT(num_classes)


def evaluate(model, dataloader, device, logger):
    """
    Evaluate model on dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run on
        logger: Logger instance
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    logger.info("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate accuracy
    accuracy_metrics = compute_accuracy(all_predictions, all_targets, topk=(1, 5))
    
    logger.info(f"Top-1 Accuracy: {accuracy_metrics['top1_accuracy']:.2f}%")
    logger.info(f"Top-5 Accuracy: {accuracy_metrics['top5_accuracy']:.2f}%")
    
    return accuracy_metrics


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.device:
        config.eval.device = args.device
    if args.batch_size:
        config.eval.batch_size = args.batch_size
    
    # Setup logger
    log_file = Path(config.logging.log_dir) / f"{config.experiment_name}_eval.log"
    logger = setup_logger(
        name="rajni_eval",
        log_file=str(log_file),
        log_level=getattr(__import__('logging'), config.logging.log_level)
    )
    
    logger.info(f"Starting RAJNI-ViT evaluation: {config.experiment_name}")
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Set device
    device = torch.device(config.eval.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(config.seed)
    
    # Create base ViT model (placeholder - use timm in practice)
    logger.info(f"Loading model: {config.model.name}")
    base_vit = create_dummy_vit(config.model.num_classes)
    
    # Wrap with RAJNI pruning
    logger.info("Wrapping model with RAJNI pruning...")
    model = RAJNIViT(
        vit_model=base_vit,
        pruning_ratio=config.pruning.pruning_ratio,
        num_pruning_layers=config.pruning.num_pruning_layers,
        keep_cls_token=config.pruning.keep_cls_token,
        importance_metric=config.pruning.importance_metric
    )
    model = model.to(device)
    
    logger.info(f"RAJNI Pruning Configuration:")
    logger.info(f"  - Pruning ratio: {config.pruning.pruning_ratio}")
    logger.info(f"  - Pruning layers: {config.pruning.num_pruning_layers}")
    logger.info(f"  - Importance metric: {config.pruning.importance_metric}")
    
    # Create dataloader (placeholder - use real dataset in practice)
    logger.info("Creating dataloader...")
    dataloader = create_dummy_dataloader(
        batch_size=config.eval.batch_size,
        image_size=config.model.image_size,
        num_batches=20
    )
    
    # Evaluate model
    metrics = evaluate(model, dataloader, device, logger)
    
    # Measure throughput
    logger.info("Measuring throughput...")
    sample_input = torch.randn(
        config.eval.batch_size,
        3,
        config.model.image_size,
        config.model.image_size
    ).to(device)
    
    throughput_metrics = calculate_throughput(
        model,
        sample_input,
        num_iterations=50,
        warmup_iterations=10
    )
    
    logger.info(f"Throughput: {throughput_metrics['throughput_imgs_per_sec']:.2f} imgs/sec")
    logger.info(f"Latency: {throughput_metrics['latency_sec']*1000:.2f} ms")
    
    # Get pruning statistics
    stats = model.get_stats()
    logger.info(f"Pruning Statistics: {stats}")
    
    logger.info("Evaluation completed!")
    
    return metrics


if __name__ == "__main__":
    main()
