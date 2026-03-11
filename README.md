# RAJNI-ViT

**Rank-Adaptive Jacobian Neuronal Importance for Vision Transformers** — dynamic token pruning for accelerated ViT inference with minimal accuracy loss.

## Installation

```bash
pip install torch timm
```

## Usage

### Python API

```python
import torch
import timm
from rajni import RAJNIViTWrapper

base = timm.create_model("vit_base_patch16_224", pretrained=True)

schedule = {
    3: {"keep_ratio": 0.88, "update": True},
    4: {"keep_ratio": 0.88, "update": True},
    7: {"keep_ratio": 0.80, "update": True},
    8: {"keep_ratio": 0.72, "update": True},
}

model = RAJNIViTWrapper(base, schedule)
model.cuda().eval()

x = torch.randn(1, 3, 224, 224, device="cuda")
with torch.no_grad():
    out = model(x)

print(out.shape)              # classification logits
print(model.get_last_stats()) # per-layer token counts
```

### CLI Evaluation

Evaluate on an ImageNet-style validation set:

```bash
python -m rajni.run \
  --data_path /path/to/val \
  --schedule schedule.json \
  --model vit_base_patch16_224 \
  --batch_size 256 \
  --device cuda \
  --warmup 5 \
  --max_batches 100 \
  --compare_base
```

| Flag | Description |
|------|-------------|
| `--data_path` | Path to the dataset root (required) |
| `--schedule` | Path to a JSON pruning schedule file (required) |
| `--model` | Any timm ViT model name (default: `vit_base_patch16_224`) |
| `--batch_size` | Evaluation batch size (default: `256`) |
| `--device` | `cuda` or `cpu` (default: `cuda`) |
| `--warmup` | Warmup batches before timing (default: `5`) |
| `--max_batches` | Limit the number of batches evaluated |
| `--compare_base` | Also benchmark the unpruned base model |

### Programmatic Evaluation

```python
from rajni import evaluate_model

accuracy, throughput = evaluate_model(
    model=model,
    dataloader=val_loader,
    device="cuda",
    max_batches=None,
    warmup=5,
)
```

## Pruning Schedule Format

The schedule maps transformer block indices to pruning configurations:

```json
{
  "3": { "keep_ratio": 0.95, "update": false },
  "5": { "keep_ratio": 0.85, "update": true }
}
```

| Field | Description |
|-------|-------------|
| `keep_ratio` | Fraction of tokens retained at this block (e.g., `0.85` keeps 85%) |
| `update` | Recompute importance scores at this block (`true`/`false`) |
