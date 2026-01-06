# RAJNI-ViT

Rank-Adaptive Jacobian Neuronal Importance for Vision Transformers

## Installation

```bash
pip install torch timm
```

## Quick Start

```python
import torch
import timm
from rajni import RAJNIViTWrapper

# Create base ViT model
base = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
)

# Define pruning schedule
schedule = {
    3: {"keep_ratio": 0.88, "update": True},
    4: {"keep_ratio": 0.88, "update": True},
    7: {"keep_ratio": 0.8, "update": True},
    8: {"keep_ratio": 0.72, "update": True},
}

# Wrap with RAJNI
model = RAJNIViTWrapper(base, schedule)
model.cuda().eval()

# Test inference
x = torch.randn(1, 3, 224, 224, device="cuda")
with torch.no_grad():
    y = model(x)

print(y.shape)
print(model.get_last_stats())
```

## Evaluation

### Using run.py

```bash
python -m rajni.run \
  --data_path ../../Downloads/val \  #Path to dataset
  --model vit_base_patch16_224 \     
  --batch_size 256 \                 
  --schedule schedule.json \    #json file location with schedule
  --compare_base \           #Use if want a comparison with base model
  --max_batches 100 \
  --warmup 5 \
  --device cuda
```

### Programmatic Evaluation

```python
from rajni import evaluate_model
# Evaluate on validation set
acc, throughput = evaluate_model(
    model=model,
    dataloader=val_loader,
    device="cuda",
    max_batches=None,  # Use None for full dataset
    warmup=50
)
```

## Pruning Schedule

The pruning schedule is a dictionary where:
- **Key**: Transformer block index
- **Value**: Configuration dict with:
  - `keep_ratio`: Fraction of tokens to keep (e.g., 0.88 = keep 88%)
  - `update`: Whether to update importance scores dynamically
