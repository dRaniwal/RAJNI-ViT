# RAJNI-ViT

Rank-Adaptive Jacobian Neuronal Importance for Vision Transformers

**Fully Dynamic Scheduling** — No fixed pruning schedules. Token pruning adapts per-batch based on layer difficulty using q-norm exponential scheduling.

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

# Wrap with RAJNI (fully dynamic scheduling)
model = RAJNIViTWrapper(
    base,
    percentile=0.75,     # q-norm percentile for difficulty
    kr_min=0.60,         # minimum keep ratio
    gamma=2.5,           # exponential decay rate
    skip_layers=(10, 11) # layers to skip pruning
)
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
  --data_path ../../Downloads/val \
  --model vit_base_patch16_224 \
  --batch_size 256 \
  --compare_base \
  --max_batches 100 \
  --warmup 5 \
  --device cuda \
  --percentile 0.75 \
  --kr_min 0.60 \
  --gamma 2.5 \
  --skip_layers 10 11
```

Optionally, save dynamic parameters in a JSON file:

```json
{
  "percentile": 0.75,
  "kr_min": 0.60,
  "gamma": 2.5,
  "skip_layers": [10, 11]
}
```

Then run:

```bash
python -m rajni.run \
  --data_path ../../Downloads/val \
  --model vit_base_patch16_224 \
  --batch_size 256 \
  --schedule schedule.json \
  --compare_base \
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

## Dynamic Scheduling

RAJNI uses **q-norm exponential scheduling** to compute keep ratios at runtime:

1. **Importance**: `CLS-attention × sigmoid(|V|)` 
2. **Layer Difficulty**: `D_l = mean(max(q - log(score), 0)) / |q|`
   - `q = percentile(log_scores, p)`
3. **Keep Ratio**: `kr = max(kr_min, exp(-gamma × D_l))`

### Parameters

- `percentile`: Quantile for difficulty computation (default: 0.75)
- `kr_min`: Minimum keep ratio to prevent over-pruning (default: 0.60)
- `gamma`: Controls sensitivity to difficulty (default: 2.5)
- `skip_layers`: Layers that bypass pruning entirely (default: [10, 11])
