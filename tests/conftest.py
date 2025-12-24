"""
Pytest configuration and shared fixtures.
"""
import pytest
import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def device():
    """Return the best available device."""
    import torch
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture(scope="session")
def dummy_batch():
    """Create a dummy batch of images."""
    import torch
    return torch.randn(4, 3, 224, 224)


@pytest.fixture(scope="module")
def vit_tiny():
    """Create a ViT-Tiny model for testing."""
    try:
        import timm
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
        return model
    except ImportError:
        pytest.skip("timm not installed")


@pytest.fixture(scope="module")
def rajni_model(vit_tiny):
    """Create a RAJNI-wrapped ViT model."""
    from rajni import AdaptiveJacobianPrunedViT
    return AdaptiveJacobianPrunedViT(vit_tiny, gamma=0.01, min_tokens=16)
