"""
Diagnostic script to verify FLOPs and throughput measurements.

Run this on Kaggle/Colab to sanity-check your measurements.

Usage:
    python scripts/diagnostic.py
    
Or in Kaggle notebook:
    %run scripts/diagnostic.py
"""
import torch
import timm
import time
import sys
import os

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 60)
    print("RAJNI Diagnostic: FLOPs and Throughput Verification")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    if device != "cuda":
        print("\n⚠️  WARNING: Running on CPU. Timing will be less accurate.")
        print("   For proper benchmarking, use a GPU.")
    
    # Create model
    print("\n[1] Loading ViT-Base...")
    base_model = timm.create_model('vit_base_patch16_224', pretrained=False)
    base_model.to(device).eval()
    
    # Import RAJNI
    from rajni import AdaptiveJacobianPrunedViT
    from evaluation.flops import flops_reduction
    
    # Wrap with RAJNI
    gamma = 0.01
    print(f"[2] Creating RAJNI model (gamma={gamma})...")
    rajni_model = AdaptiveJacobianPrunedViT(base_model, gamma=gamma)
    rajni_model.to(device).eval()
    
    # Create dummy input
    batch_size = 32
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # ============================================
    # PART A: Verify token counts
    # ============================================
    print("\n" + "=" * 60)
    print("PART A: Token Counts per Layer")
    print("=" * 60)
    
    with torch.no_grad():
        _ = rajni_model(x)
    
    stats = rajni_model.get_last_stats()
    token_counts = stats["token_counts"]
    
    print(f"\nLayer-by-layer token counts:")
    total_tokens_rajni = 0
    total_tokens_baseline = 0
    initial_tokens = token_counts[0]
    
    for i, count in enumerate(token_counts):
        saved = initial_tokens - count
        pct = 100 * saved / initial_tokens if initial_tokens > 0 else 0
        total_tokens_rajni += count
        total_tokens_baseline += initial_tokens
        print(f"  Layer {i+1:2d}: {count:3d} tokens (saved {saved:3d}, {pct:5.1f}%)")
    
    overall_token_reduction = 100 * (1 - total_tokens_rajni / total_tokens_baseline)
    print(f"\nTotal tokens processed: {total_tokens_rajni} / {total_tokens_baseline}")
    print(f"Overall token reduction: {overall_token_reduction:.1f}%")
    
    # ============================================
    # PART B: Verify FLOPs calculation
    # ============================================
    print("\n" + "=" * 60)
    print("PART B: FLOPs Calculation")
    print("=" * 60)
    
    flops = flops_reduction(base_model, stats)
    print(f"\nBaseline:  {flops['baseline_GFLOPs']:.2f} GFLOPs")
    print(f"RAJNI:     {flops['rajni_GFLOPs']:.2f} GFLOPs")
    print(f"Reduction: {flops['reduction_percent']:.1f}%")
    
    # Sanity check
    print(f"\nSanity check:")
    print(f"  Token reduction: {overall_token_reduction:.1f}%")
    print(f"  FLOPs reduction: {flops['reduction_percent']:.1f}%")
    
    if flops['reduction_percent'] < overall_token_reduction * 0.5:
        print("  ⚠️  FLOPs reduction seems LOW compared to token reduction")
    elif flops['reduction_percent'] > overall_token_reduction * 2:
        print("  ⚠️  FLOPs reduction seems HIGH compared to token reduction")
    else:
        print("  ✓ FLOPs reduction is consistent with token reduction")
    
    # ============================================
    # PART C: Throughput measurement (careful!)
    # ============================================
    print("\n" + "=" * 60)
    print("PART C: Throughput Measurement (Fair Comparison)")
    print("=" * 60)
    
    num_iterations = 50
    warmup_iters = 20
    
    # CRITICAL: Create a FRESH baseline model for fair comparison
    print("\n[!] Creating FRESH baseline model...")
    baseline_fresh = timm.create_model('vit_base_patch16_224', pretrained=False)
    baseline_fresh.to(device).eval()
    
    # Warmup BOTH models SEPARATELY (this is key!)
    print(f"\nWarming up BASELINE ({warmup_iters} iterations)...")
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = baseline_fresh(x)
    if device == "cuda":
        torch.cuda.synchronize()
    
    print(f"Warming up RAJNI ({warmup_iters} iterations)...")
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = rajni_model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Use CUDA events for accurate timing
    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    
    # Time BASELINE FIRST
    print(f"\nTiming BASELINE ({num_iterations} iterations)...")
    if device == "cuda":
        torch.cuda.synchronize()
        start_event.record()
    else:
        start = time.time()
        
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = baseline_fresh(x)
    
    if device == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        baseline_time = start_event.elapsed_time(end_event) / 1000.0
    else:
        baseline_time = time.time() - start
    baseline_throughput = (num_iterations * batch_size) / baseline_time
    
    # Time RAJNI SECOND
    print(f"Timing RAJNI ({num_iterations} iterations)...")
    if device == "cuda":
        torch.cuda.synchronize()
        start_event.record()
    else:
        start = time.time()
        
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = rajni_model(x)
    
    if device == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        rajni_time = start_event.elapsed_time(end_event) / 1000.0
    else:
        rajni_time = time.time() - start
    rajni_throughput = (num_iterations * batch_size) / rajni_time
    
    speedup = rajni_throughput / baseline_throughput
    
    print(f"\n┌────────────────────────────────────────┐")
    print(f"│  TIMING RESULTS                        │")
    print(f"├────────────────────────────────────────┤")
    print(f"│  Baseline:  {baseline_throughput:7.1f} img/s ({baseline_time:.2f}s)    │")
    print(f"│  RAJNI:     {rajni_throughput:7.1f} img/s ({rajni_time:.2f}s)    │")
    print(f"│  Speedup:   {speedup:7.2f}x                    │")
    print(f"└────────────────────────────────────────┘")
    
    # ============================================
    # PART D: Consistency check
    # ============================================
    print("\n" + "=" * 60)
    print("PART D: Consistency Analysis")
    print("=" * 60)
    
    # Expected speedup from FLOPs reduction
    flops_speedup = 1.0 / (1.0 - flops['reduction_percent'] / 100)
    
    print(f"\nFLOPs reduction:      {flops['reduction_percent']:.1f}%")
    print(f"Theoretical speedup:  {flops_speedup:.2f}x (from FLOPs)")
    print(f"Measured speedup:     {speedup:.2f}x")
    
    # Analysis
    ratio = speedup / flops_speedup if flops_speedup > 1 else speedup
    
    print(f"\nAnalysis:")
    if ratio > 2.0:
        print("  ❌ ERROR: Measured speedup >> theoretical!")
        print("     This indicates a measurement problem.")
        print("     Possible causes:")
        print("     - Baseline wasn't properly warmed up")
        print("     - Different models being compared")
        print("     - CUDA caching/memory effects")
        print("     - DataLoader bottleneck (not GPU-bound)")
    elif ratio > 1.5:
        print("  ⚠️  WARNING: Measured speedup > theoretical")
        print("     Possible causes:")
        print("     - Memory bandwidth improvements (fewer tokens = better cache)")
        print("     - Measurement variance")
    elif speedup < 1.0:
        print("  ⚠️  WARNING: RAJNI is SLOWER than baseline!")
        print("     With low gamma, pruning overhead may exceed savings.")
        print("     Try increasing gamma (0.02, 0.05, 0.1)")
    elif ratio < 0.7:
        print("  ⚠️  NOTE: Speedup < theoretical")
        print("     This is NORMAL - real speedup is usually less than")
        print("     theoretical due to:")
        print("     - Token selection overhead")
        print("     - Memory access patterns")
        print("     - Kernel launch overhead")
    else:
        print("  ✓ Speedup is consistent with FLOPs reduction!")
    
    print("\n" + "=" * 60)
    print("EXPECTED RELATIONSHIPS:")
    print("=" * 60)
    print("""
For gamma=0.01 on ViT-Base, you should see approximately:

  Token reduction:  ~5-15%
  FLOPs reduction:  ~8-20%  
  Real speedup:     ~1.05-1.15x

If you're seeing 3x speedup, something is WRONG with measurement.

Common fixes:
1. Warmup BOTH models before timing
2. Use the same batch size for both
3. Create a FRESH baseline model (don't reuse the wrapped one)
4. Use CUDA events instead of time.time()
""")
    
    print("=" * 60)
    print("Diagnostic complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
