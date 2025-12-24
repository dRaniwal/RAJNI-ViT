# RAJNI-ViT

<p align="center">
  <strong>Relative Adaptive Jacobian-based Neuronal Importance</strong><br>
  <em>Efficient Vision Transformers via Adaptive Token Pruning</em>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#method">Method</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

RAJNI is an inference-time token pruning method for Vision Transformers. It uses a first-order approximation of the Jacobian to estimate how much each patch token contributes to the CLS token—without any backpropagation or training.

**Key features:**
- 🚀 **No training required** – works with any pretrained ViT
- 📉 **Adaptive pruning** – adjusts pruning rate based on layer dynamics
- ⚡ **Real speedups** – reduces FLOPs by 30-50% with <1% accuracy drop
- 🔧 **Drop-in wrapper** – one line to add pruning to your model

## Installation

```bash
# Clone the repository
git clone https://github.com/dRaniwal/RAJNI-ViT.git
cd RAJNI-ViT

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 1.12
- timm ≥ 0.9.0

## Quick Start

```python
import timm
from rajni import AdaptiveJacobianPrunedViT

# Load any pretrained ViT
base_model = timm.create_model('vit_base_patch16_224', pretrained=True)

# Wrap with RAJNI
model = AdaptiveJacobianPrunedViT(
    base_model,
    gamma=0.02,      # Pruning intensity (higher = more aggressive)
    min_tokens=16,   # Minimum tokens to keep
)

# Use as normal
model.eval()
logits = model(images)

# Access pruning statistics
stats = model.get_last_stats()
print(f"Token counts per layer: {stats['token_counts']}")
```

## Method

RAJNI approximates the gradient of the CLS token with respect to each patch token using attention weights and value vector norms:

$$\text{importance}_j \approx \sum_{h=1}^{H} |A_{0,j}^{(h)}| \cdot \|V_j^{(h)}\|$$

where $A_{0,j}^{(h)}$ is the attention from CLS to patch $j$ in head $h$, and $V_j^{(h)}$ is the corresponding value vector.

The pruning budget at each layer is determined adaptively:

$$\text{keep\_ratio} = \min\left(1, \left(\rho \cdot \frac{\eta}{\eta_{\text{prev}}}\right)^{-\gamma}\right)$$

where:
- $\rho$ = CLS sensitivity (how much CLS attends to patches)
- $\eta$ = total importance mass in the current layer
- $\gamma$ = user-controlled pruning intensity

## Project Structure

```
RAJNI-ViT/
├── rajni/                    # Core library
│   ├── __init__.py           # Main exports
│   ├── model.py              # AdaptiveJacobianPrunedViT wrapper
│   ├── pruning.py            # Pure pruning algorithms
│   └── utils.py              # Utilities
│
├── evaluation/               # Evaluation tools
│   ├── benchmark.py          # Accuracy + throughput
│   ├── baseline.py           # Baseline comparison
│   ├── flops.py              # FLOPs analysis
│   └── visualize.py          # Pruning visualization
│
├── examples/                 # Usage examples
│   ├── run_imagenet.py       # ImageNet evaluation
│   ├── run_cifar.py          # CIFAR evaluation
│   └── demo.ipynb            # Interactive demo
│
├── scripts/                  # Experiment scripts
│   ├── sweep_gamma.sh        # Hyperparameter sweep
│   ├── benchmark_imagenet.sh # Standard benchmark
│   └── compare_models.sh     # Model comparison
│
└── tests/                    # Unit tests
    ├── test_model.py
    └── test_flops.py
```

## Experiments

### Benchmark on ImageNet

```bash
# Standard evaluation
python examples/run_imagenet.py \
    --data /path/to/imagenet \
    --model vit_base_patch16_224 \
    --gamma 0.02

# Or use the shell script
./scripts/benchmark_imagenet.sh /path/to/imagenet
```

### Hyperparameter Sweep

```bash
# Sweep over gamma values
./scripts/sweep_gamma.sh /path/to/imagenet

# Compare different model sizes
./scripts/compare_models.sh /path/to/imagenet
```

### Expected Results (ViT-B/16 on ImageNet-1K)

| Method | Top-1 Acc | FLOPs | Reduction |
|--------|-----------|-------|-----------|
| Baseline | 81.8% | 17.6G | – |
| RAJNI (γ=0.01) | 81.5% | 14.1G | 20% |
| RAJNI (γ=0.02) | 81.2% | 12.3G | 30% |
| RAJNI (γ=0.05) | 80.4% | 9.7G | 45% |

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v

# With coverage
pytest tests/ --cov=rajni --cov-report=term-missing
```

## Visualization

```python
from evaluation.visualize import visualise_pruning

# Visualize which patches get pruned
visualise_pruning(model, dataloader, device='cuda')
```

## Citation

If you find RAJNI useful in your research, please cite:

```bibtex
@inproceedings{raniwal2025rajni,
  title={RAJNI: Relative Adaptive Jacobian-based Neuronal Importance for Efficient Vision Transformers},
  author={Raniwal, Dhairya},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [timm](https://github.com/huggingface/pytorch-image-models) for pretrained Vision Transformers
- [ToMe](https://github.com/facebookresearch/ToMe) and [DynamicViT](https://github.com/raoyongming/DynamicViT) for inspiring efficient ViT research
