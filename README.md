# RAJNI-ViT

Rank-Adaptive Jacobian Neuronal Importance for Vision Transformers

**Fully Dynamic Scheduling** — No fixed pruning schedules. Each layer's dispersion statistic $D_l$ decides at runtime whether to skip pruning (warmup phase, $D_l < \tau_{\text{warmup}}$) or prune with an exponentially-decaying keep ratio (Phase 2).

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
    percentile=0.75,     # kept for API compatibility; see note below
    kr_min=0.60,         # minimum keep ratio
    gamma=2.5,           # exponential decay rate
    tau_warmup=0.10,     # dispersion threshold: below this, layer skips pruning
)
model.cuda().eval()

# Test inference
x = torch.randn(1, 3, 224, 224, device="cuda")
with torch.no_grad():
    y = model(x)

print(y.shape)
print(model.last_stats)
```

> **Note:** `percentile` is retained in the constructor for API stability, but the
> forward pass currently uses a fixed Gaussian z-score approximation
> (`mu + 0.675 * sigma`, i.e. the 75th percentile under a normal assumption) to
> estimate the log-score threshold, rather than reading `percentile` at runtime.
> If you need a different percentile to actually take effect, this is the line
> to change in `rajni/wrapper/attention.py`.

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
  --tau_warmup 0.10
```

Optionally, save dynamic parameters in a JSON file:

```json
{
  "percentile": 0.75,
  "kr_min": 0.60,
  "gamma": 2.5,
  "tau_warmup": 0.10
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

## Best Hyperparameters (Reproducing Paper Results)

All operating points below were found by a TPE (Optuna) search over `(tau_warmup, gamma)`
per backbone, with `kr_min=0.60` and `percentile=0.75` held fixed. These are the exact
values behind the representative operating points reported in the paper; the full sweep
(23 non-dominated points for ViT-Base, 17 for DeiT-S) is in the paper's Appendix A.1.

### ViT-Base/16 — `timm` model `vit_base_patch16_224` (baseline: 81.10% top-1)

| Operating point | `tau_warmup` | `gamma` | Speedup | Top-1 (%) | ΔAcc (pp) |
|---|---|---|---|---|---|
| Conservative | `0.1008` | `0.728` | 1.08× | 81.06 | −0.04 |
| Balanced     | `0.1051` | `0.852` | 1.24× | 80.87 | −0.23 |
| Aggressive   | `0.0716` | `0.768` | 1.41× | 80.16–80.26 | −0.85 to −0.95 |

### DeiT-S/16 — `timm` model `deit_small_patch16_224` (baseline: 79.73% top-1)

| Operating point | `tau_warmup` | `gamma` | Speedup | Top-1 (%) | ΔAcc (pp) |
|---|---|---|---|---|---|
| Conservative | `0.1274` | `0.807` | 1.07× | 79.64 | −0.08 |
| Aggressive   | `0.0700` | `0.993` | 1.36× | 78.75 | −0.97 |

To reproduce, e.g., the ViT-Base **balanced** point:

```bash
python -m rajni.run \
  --data_path /path/to/imagenet/val \
  --model vit_base_patch16_224 \
  --batch_size 256 \
  --compare_base \
  --device cuda \
  --percentile 0.75 \
  --kr_min 0.60 \
  --gamma 0.852 \
  --tau_warmup 0.1051
```

or drop it into a schedule file:

```json
{
  "percentile": 0.75,
  "kr_min": 0.60,
  "gamma": 0.852,
  "tau_warmup": 0.1051
}
```

```bash
python -m rajni.run \
  --data_path /path/to/imagenet/val \
  --model vit_base_patch16_224 \
  --batch_size 256 \
  --schedule schedule.json \
  --compare_base \
  --device cuda
```

> These numbers were measured on a single NVIDIA Tesla P100, FP32, batch size 256, full
> ImageNet-1K validation set (50,000 images) — see the paper's Experimental Setup and
> Appendix A.3 (sourcing disclosure) for the full protocol. Reproduced numbers on other
> hardware may vary slightly in throughput, though accuracy should match closely since
> pruning decisions are deterministic given `(tau_warmup, gamma, kr_min)`.

## Dynamic Scheduling

RAJNI uses **dispersion-based exponential scheduling** to compute keep ratios per layer, per batch, at runtime — there is no fixed per-layer schedule and no hardcoded skip list.

1. **Importance**: `CLS-attention × sigmoid(|V|)`, computed from the block's own QKV
   projection (`rajni/wrapper/importance.py`).
2. **Layer Dispersion** (`D_l`): how concentrated importance is on a few tokens vs.
   spread out.
   - `log_scores = log(importance[:, 1:] + eps)`
   - `q ≈ mean(log_scores) + 0.675 * std(log_scores)` (Gaussian approximation of the
     75th percentile)
   - `D_l = mean(max(q - log_scores, 0)) / |q|`
3. **Phase gate**: if `D_l < tau_warmup`, the layer is still gathering diffuse,
   undifferentiated features — pruning is skipped entirely for that layer on that
   batch (full attention, no gather/scatter overhead). Otherwise the layer has
   reached semantic saturation and is pruned.
4. **Keep Ratio** (Phase 2 only): `kr = max(kr_min, exp(-gamma * D_l))`, applied via
   top-k selection on the raw importance scores (no masking — pruned tokens are
   physically removed from the sequence for the rest of that block).

Because the gate is evaluated fresh per layer per batch from `D_l`, easy/uniform
inputs and later, more class-differentiated layers tend to prune more aggressively,
while early layers or visually complex inputs naturally fall back to full attention.

### Parameters

- `percentile`: reserved for future use — the 75th-percentile threshold in step 2
  is currently a fixed Gaussian approximation, not read from this argument (see the
  note in Quick Start above).
- `kr_min`: minimum keep ratio to prevent over-pruning (default: 0.60)
- `gamma`: controls how sharply keep ratio drops off with dispersion (default: 2.5)
- `tau_warmup`: dispersion threshold below which a layer skips pruning entirely
  (default: 0.10) — replaces the old fixed `skip_layers` list
- `compile_blocks`: if `True`, wraps each block's MLP in `torch.compile(mode="reduce-overhead")`
  for extra throughput (default: `False`)
