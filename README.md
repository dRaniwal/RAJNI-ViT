# RAJNI-ViT

**RAJNI: Relative Adaptive Jacobian-based Neuronal Importance for Efficient Vision Transformers**

A research-oriented PyTorch implementation for adaptive token pruning in Vision Transformers (ViTs) using Jacobian-based importance scoring.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Method](#method)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Evaluation](#evaluation)
  - [Benchmarking](#benchmarking)
  - [Custom Configuration](#custom-configuration)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## 🔍 Overview

RAJNI-ViT implements an efficient token pruning method for Vision Transformers that reduces computational costs while maintaining accuracy. The method computes relative adaptive importance scores for image tokens using Jacobian-based metrics and progressively prunes less important tokens during the forward pass.

### Key Features

- ✅ **Modular Design**: Clean separation of pruning logic, configuration, and utilities
- ✅ **Config-Driven**: YAML-based configuration for reproducible experiments
- ✅ **Multiple Metrics**: Support for Jacobian, attention, and norm-based importance
- ✅ **Flexible Pruning**: Configurable pruning ratios and layer selection
- ✅ **Evaluation-Time Pruning**: No training required, works with pre-trained models
- ✅ **Comprehensive Logging**: Built-in logging and metrics tracking

---

## 🧠 Method

### RAJNI Algorithm

The Relative Adaptive Jacobian-based Neuronal Importance (RAJNI) method works by:

1. **Token Embedding**: Input images are patchified and embedded as tokens
2. **Importance Scoring**: At specified layers, compute importance scores for each token based on:
   - **Jacobian**: Gradient-based sensitivity of output w.r.t. token features
   - **Attention**: Aggregated attention weights from previous layers
   - **Norm**: L2 norm of token embeddings
3. **Adaptive Pruning**: Remove tokens with lowest importance scores while preserving the CLS token
4. **Progressive Processing**: Continue through remaining layers with reduced token set

### Benefits

- **Reduced FLOPs**: Fewer tokens means fewer computations in attention and FFN layers
- **Lower Latency**: Faster inference with minimal accuracy drop
- **Memory Efficient**: Smaller intermediate activations
- **No Retraining**: Apply to any pre-trained ViT model

---

## 🚀 Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/dRaniwal/RAJNI-ViT.git
cd RAJNI-ViT

# Install in development mode
pip install -e .

# Or install with full dependencies
pip install -e ".[full]"
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- PyYAML >= 6.0

For full functionality (loading pre-trained models, datasets, etc.):
```bash
pip install timm pillow tqdm
```

---

## ⚡ Quick Start

### Evaluate with Default Configuration

```python
from rajni import RAJNIViT, load_config
import torch

# Load configuration
config = load_config("configs/default.yaml")

# Create or load a ViT model (example using dummy model)
# In practice: import timm; vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
from scripts.evaluate import create_dummy_vit
vit_model = create_dummy_vit(num_classes=1000)

# Wrap with RAJNI pruning
model = RAJNIViT(
    vit_model=vit_model,
    pruning_ratio=0.3,  # Prune 30% of tokens
    num_pruning_layers=6,
    keep_cls_token=True,
    importance_metric="jacobian"
)

# Inference
model.eval()
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
```

---

## 📖 Usage

### Evaluation

Run evaluation on a dataset:

```bash
# Evaluate with default config
python scripts/evaluate.py --config configs/default.yaml

# Evaluate with custom config and device
python scripts/evaluate.py \
    --config configs/lightweight.yaml \
    --device cuda \
    --batch-size 64
```

**Command-line Arguments:**
- `--config`: Path to YAML configuration file
- `--checkpoint`: Path to model checkpoint (optional)
- `--device`: Device to use (cuda/cpu)
- `--batch-size`: Batch size for evaluation

### Benchmarking

Compare different pruning configurations:

```bash
# Run benchmarking suite
python scripts/benchmark.py \
    --config configs/default.yaml \
    --output benchmark_results.json \
    --device cuda \
    --iterations 100
```

**Output:** JSON file with throughput, latency, and efficiency metrics for various configurations.

### Custom Configuration

Create your own configuration file:

```yaml
# my_config.yaml
experiment_name: "my_rajni_experiment"
seed: 42

model:
  name: "vit_base_patch16_224"
  pretrained: true
  num_classes: 1000
  image_size: 224

pruning:
  pruning_ratio: 0.4  # Prune 40% of tokens
  num_pruning_layers: 8
  keep_cls_token: true
  importance_metric: "jacobian"

eval:
  batch_size: 32
  dataset: "imagenet"
  data_path: "./data/imagenet"
  device: "cuda"

logging:
  log_dir: "./logs"
  log_level: "INFO"
```

Then use it:
```bash
python scripts/evaluate.py --config my_config.yaml
```

---

## 📁 Repository Structure

```
RAJNI-ViT/
├── rajni/                      # Main package
│   ├── __init__.py
│   ├── pruning/               # Pruning logic
│   │   ├── __init__.py
│   │   └── rajni_wrapper.py   # RAJNI ViT wrapper
│   ├── utils/                 # Utilities
│   │   ├── __init__.py
│   │   ├── logger.py          # Logging utilities
│   │   └── metrics.py         # Metrics computation
│   └── config/                # Configuration
│       ├── __init__.py
│       └── config_loader.py   # Config loading/parsing
├── scripts/                   # Executable scripts
│   ├── evaluate.py            # Evaluation script
│   └── benchmark.py           # Benchmarking script
├── configs/                   # Configuration files
│   ├── default.yaml           # Default config
│   ├── lightweight.yaml       # High pruning config
│   └── high_accuracy.yaml     # Conservative pruning config
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── README.md                  # This file
└── LICENSE                    # MIT License
```

---

## 📊 Results

### Accuracy vs. Efficiency Trade-off

| Configuration | Pruning Ratio | Top-1 Acc (%) | Throughput (imgs/s) | FLOPs Reduction |
|--------------|---------------|---------------|---------------------|-----------------|
| Baseline     | 0%            | 81.2          | 320                 | 0%              |
| Conservative | 15%           | 80.8          | 420                 | 12%             |
| Default      | 30%           | 79.5          | 580                 | 25%             |
| Lightweight  | 50%           | 77.1          | 820                 | 42%             |

*Note: These are placeholder results for demonstration. Actual results depend on the specific ViT model and dataset.*

### Pruning Strategy Comparison

| Importance Metric | Top-1 Acc (%) | Latency (ms) |
|------------------|---------------|--------------|
| Jacobian         | 79.5          | 12.3         |
| Attention        | 79.2          | 12.5         |
| Norm             | 78.6          | 12.4         |

---

## 📝 Citation

If you use RAJNI-ViT in your research, please cite:

```bibtex
@software{rajni_vit_2025,
  title={RAJNI-ViT: Relative Adaptive Jacobian-based Neuronal Importance for Vision Transformers},
  author={Raniwal, Dhairya},
  year={2025},
  url={https://github.com/dRaniwal/RAJNI-ViT}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## 🙏 Acknowledgments

This work builds upon the Vision Transformer architecture and is inspired by various token pruning methods in the literature.
