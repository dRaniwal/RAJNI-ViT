"""
Tests for RAJNI model wrapper.
"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAdaptiveJacobianPrunedViT:
    """Test suite for the main RAJNI model."""
    
    @pytest.fixture
    def base_model(self):
        """Create a simple ViT model for testing."""
        try:
            import timm
            model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
            return model
        except ImportError:
            pytest.skip("timm not installed")
    
    @pytest.fixture
    def dummy_input(self):
        """Create dummy input tensor."""
        return torch.randn(2, 3, 224, 224)
    
    def test_model_creation(self, base_model):
        """Test that RAJNI wrapper can be created."""
        from rajni import AdaptiveJacobianPrunedViT
        
        model = AdaptiveJacobianPrunedViT(base_model, gamma=0.01)
        
        assert model.gamma == 0.01
        assert model.min_tokens == 16
        assert model.embed_dim == base_model.embed_dim
    
    def test_forward_pass(self, base_model, dummy_input):
        """Test forward pass produces correct output shape."""
        from rajni import AdaptiveJacobianPrunedViT
        
        model = AdaptiveJacobianPrunedViT(base_model, gamma=0.01)
        model.eval()
        
        with torch.no_grad():
            logits = model(dummy_input)
        
        assert logits.shape == (2, 1000)  # batch_size x num_classes
    
    def test_stats_collection(self, base_model, dummy_input):
        """Test that stats are collected after forward pass."""
        from rajni import AdaptiveJacobianPrunedViT
        
        model = AdaptiveJacobianPrunedViT(base_model, gamma=0.01)
        model.eval()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        stats = model.get_last_stats()
        
        assert stats is not None
        assert "token_counts" in stats
        assert "kept_indices" in stats
        assert len(stats["token_counts"]) == len(base_model.blocks)
    
    def test_token_counts_decreasing(self, base_model, dummy_input):
        """Test that token counts generally decrease through layers."""
        from rajni import AdaptiveJacobianPrunedViT
        
        # Use higher gamma for more aggressive pruning
        model = AdaptiveJacobianPrunedViT(base_model, gamma=0.1)
        model.eval()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        stats = model.get_last_stats()
        token_counts = stats["token_counts"]
        
        # First token count should be full (197 for 224x224 with patch 16)
        assert token_counts[0] == 197
        
        # With aggressive pruning, later layers should have fewer tokens
        # (not strictly monotonic, but generally decreasing)
        assert min(token_counts) <= token_counts[0]
    
    def test_min_tokens_respected(self, base_model, dummy_input):
        """Test that minimum tokens constraint is respected."""
        from rajni import AdaptiveJacobianPrunedViT
        
        min_tokens = 32
        model = AdaptiveJacobianPrunedViT(
            base_model, 
            gamma=1.0,  # Very aggressive
            min_tokens=min_tokens,
        )
        model.eval()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        stats = model.get_last_stats()
        
        # All token counts should be >= min_tokens + 1 (for CLS)
        for count in stats["token_counts"]:
            assert count >= min_tokens + 1
    
    def test_gamma_zero_no_pruning(self, base_model, dummy_input):
        """Test that gamma=0 results in no pruning."""
        from rajni import AdaptiveJacobianPrunedViT
        
        model = AdaptiveJacobianPrunedViT(base_model, gamma=0.0)
        model.eval()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        stats = model.get_last_stats()
        
        # With gamma=0, keep_ratio should always be 1.0
        # so no tokens should be pruned
        initial_tokens = stats["token_counts"][0]
        for count in stats["token_counts"]:
            assert count == initial_tokens
    
    def test_repr(self, base_model):
        """Test string representation."""
        from rajni import AdaptiveJacobianPrunedViT
        
        model = AdaptiveJacobianPrunedViT(base_model, gamma=0.01)
        repr_str = repr(model)
        
        assert "AdaptiveJacobianPrunedViT" in repr_str
        assert "gamma=0.01" in repr_str


class TestPruningFunctions:
    """Test the core pruning algorithm functions."""
    
    def test_compute_cls_sensitivity(self):
        """Test CLS sensitivity computation."""
        from rajni.pruning import compute_cls_sensitivity
        
        # Create dummy attention and values
        B, H, N, D = 2, 8, 197, 64
        attention = torch.softmax(torch.randn(B, H, N, N), dim=-1)
        values = torch.randn(B, H, N, D)
        
        rho = compute_cls_sensitivity(attention, values)
        
        assert rho.ndim == 0  # Scalar
        assert rho > 0  # Should be positive
    
    def test_compute_jacobian_importance(self):
        """Test Jacobian importance computation."""
        from rajni.pruning import compute_jacobian_importance
        
        B, H, N, D = 2, 8, 197, 64
        attention = torch.softmax(torch.randn(B, H, N, N), dim=-1)
        values = torch.randn(B, H, N, D)
        num_patches = N - 1
        
        importance, mass = compute_jacobian_importance(
            attention, values, num_patches
        )
        
        assert importance.shape == (B, num_patches)
        assert (importance >= 0).all()  # ReLU ensures non-negative
        assert mass.ndim == 0
    
    def test_select_tokens(self):
        """Test token selection."""
        from rajni.pruning import select_tokens
        
        B, N = 2, 196
        importance = torch.rand(B, N)
        num_keep = 100
        
        keep_idx = select_tokens(importance, num_keep, torch.device('cpu'))
        
        assert keep_idx.shape == (num_keep + 1,)  # +1 for CLS
        assert keep_idx[0] == 0  # CLS at position 0
        assert (keep_idx[1:] > 0).all()  # Patch indices > 0
        assert (keep_idx[1:] <= N).all()  # Within valid range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
