#!/usr/bin/env python3
"""
Simple example demonstrating RAJNI-ViT usage.

This script shows how to use the RAJNI wrapper with a dummy model.
In practice, replace the dummy model with a real pre-trained ViT.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rajni import RAJNIViT, setup_logger


def main():
    """Simple RAJNI-ViT example."""
    
    # Setup logger
    logger = setup_logger(name="example", console=True)
    logger.info("RAJNI-ViT Simple Example")
    logger.info("=" * 50)
    
    # Create a dummy ViT model for demonstration
    import torch.nn as nn
    
    class DummyViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(768, 1000)
        
        def forward(self, x):
            batch_size = x.shape[0]
            features = torch.randn(batch_size, 768)
            return self.fc(features)
    
    logger.info("Creating dummy ViT model...")
    base_vit = DummyViT()
    
    # Wrap with RAJNI pruning
    logger.info("Wrapping with RAJNI pruning...")
    model = RAJNIViT(
        vit_model=base_vit,
        pruning_ratio=0.3,  # Prune 30% of tokens
        num_pruning_layers=6,
        keep_cls_token=True,
        importance_metric="jacobian"
    )
    
    logger.info("\nRAJNI Configuration:")
    logger.info(f"  - Pruning Ratio: 30%")
    logger.info(f"  - Pruning Layers: 6")
    logger.info(f"  - Keep CLS Token: Yes")
    logger.info(f"  - Importance Metric: Jacobian")
    
    # Create sample input
    logger.info("\nRunning inference on sample input...")
    model.eval()
    sample_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    
    with torch.no_grad():
        output = model(sample_input)
    
    logger.info(f"Input shape: {sample_input.shape}")
    logger.info(f"Output shape: {output.shape}")
    
    # Get statistics
    stats = model.get_stats()
    logger.info(f"\nPruning Statistics: {stats}")
    
    logger.info("\n" + "=" * 50)
    logger.info("Example completed!")
    logger.info("\nTo use with real models:")
    logger.info("  1. Install timm: pip install timm")
    logger.info("  2. Load model: vit = timm.create_model('vit_base_patch16_224', pretrained=True)")
    logger.info("  3. Wrap: rajni_vit = RAJNIViT(vit, pruning_ratio=0.3, ...)")


if __name__ == "__main__":
    main()
