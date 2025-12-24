"""
Tests for FLOPs computation.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFLOPsCounter:
    """Test suite for FLOPs counting utilities."""
    
    def test_attention_flops(self):
        """Test attention FLOPs computation."""
        from evaluation.flops import FLOPsCounter
        
        counter = FLOPsCounter(
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4.0,
            num_classes=1000,
            num_blocks=12,
        )
        
        num_tokens = 197
        attn_flops = counter.attention_flops(num_tokens)
        
        # Q, K, V projections + attention + output projection
        # Should be a positive number
        assert attn_flops > 0
        
        # Attention FLOPs should scale quadratically with tokens
        attn_flops_half = counter.attention_flops(num_tokens // 2)
        # Roughly 4x fewer FLOPs for half the tokens (quadratic)
        ratio = attn_flops / attn_flops_half
        assert 3.0 < ratio < 5.0
    
    def test_mlp_flops(self):
        """Test MLP FLOPs computation."""
        from evaluation.flops import FLOPsCounter
        
        counter = FLOPsCounter(
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4.0,
            num_classes=1000,
            num_blocks=12,
        )
        
        num_tokens = 197
        mlp_flops = counter.mlp_flops(num_tokens)
        
        # Should be positive
        assert mlp_flops > 0
        
        # MLP FLOPs should scale linearly with tokens
        mlp_flops_half = counter.mlp_flops(num_tokens // 2)
        ratio = mlp_flops / mlp_flops_half
        # Allow some tolerance due to integer division
        assert 1.8 < ratio < 2.2
    
    def test_block_flops(self):
        """Test single block FLOPs computation."""
        from evaluation.flops import FLOPsCounter
        
        counter = FLOPsCounter(
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4.0,
            num_classes=1000,
            num_blocks=12,
        )
        
        num_tokens = 197
        block_flops = counter.block_flops(num_tokens)
        attn_flops = counter.attention_flops(num_tokens)
        mlp_flops = counter.mlp_flops(num_tokens)
        
        # Block FLOPs should be sum of attention and MLP
        assert block_flops == attn_flops + mlp_flops
    
    def test_full_model_flops_uniform(self):
        """Test full model FLOPs with uniform token counts."""
        from evaluation.flops import FLOPsCounter
        
        counter = FLOPsCounter(
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4.0,
            num_classes=1000,
            num_blocks=12,
        )
        
        num_tokens = 197
        token_counts = [num_tokens] * 12
        
        total_flops = counter.total_flops(token_counts)
        
        assert total_flops > 0
        
        # Should be 12 blocks worth
        single_block = counter.block_flops(num_tokens)
        # Plus embedding and head overhead
        assert total_flops >= single_block * 12
    
    def test_flops_reduction_with_pruning(self):
        """Test that pruned model has fewer FLOPs."""
        from evaluation.flops import FLOPsCounter
        
        counter = FLOPsCounter(
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4.0,
            num_classes=1000,
            num_blocks=12,
        )
        
        # Full model
        full_counts = [197] * 12
        full_flops = counter.total_flops(full_counts)
        
        # Pruned model (decreasing tokens)
        pruned_counts = [197, 180, 160, 140, 120, 100, 80, 60, 50, 40, 35, 30]
        pruned_flops = counter.total_flops(pruned_counts)
        
        # Pruned model should use fewer FLOPs
        assert pruned_flops < full_flops
        
        # Calculate reduction
        reduction = 1.0 - (pruned_flops / full_flops)
        assert reduction > 0.2  # At least 20% reduction
    
    def test_embedding_flops(self):
        """Test embedding layer FLOPs."""
        from evaluation.flops import FLOPsCounter
        
        counter = FLOPsCounter(
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4.0,
            num_classes=1000,
            num_blocks=12,
        )
        
        embed_flops = counter.embedding_flops()
        
        # Should account for patch embedding projection
        assert embed_flops > 0
    
    def test_head_flops(self):
        """Test classification head FLOPs."""
        from evaluation.flops import FLOPsCounter
        
        counter = FLOPsCounter(
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4.0,
            num_classes=1000,
            num_blocks=12,
        )
        
        head_flops = counter.head_flops()
        
        # Linear projection: 768 * 1000 * 2
        expected = 768 * 1000 * 2
        assert head_flops == expected


class TestFLOPsFromModel:
    """Test FLOPs computation from actual models."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample RAJNI model."""
        try:
            import torch
            import timm
            from rajni import AdaptiveJacobianPrunedViT
            
            base = timm.create_model('vit_tiny_patch16_224', pretrained=False)
            model = AdaptiveJacobianPrunedViT(base, gamma=0.01)
            return model
        except ImportError:
            pytest.skip("Required packages not installed")
    
    def test_flops_from_stats(self, sample_model):
        """Test FLOPs computation from model stats."""
        import torch
        from evaluation.flops import compute_flops_from_stats
        
        # Run forward pass
        sample_model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            _ = sample_model(dummy_input)
        
        stats = sample_model.get_last_stats()
        
        # Compute FLOPs
        flops = compute_flops_from_stats(stats, sample_model.base_model)
        
        assert flops > 0
        assert isinstance(flops, (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
