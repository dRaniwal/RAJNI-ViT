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


def time_model(model, x, num_iterations=50, warmup=20, device="cuda"):
    """Time a model with proper warmup and CUDA events."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Time with CUDA events
    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
        
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0
    else:
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
        elapsed_time = time.time() - start
    
    return elapsed_time


def main():
    print("=" * 60)
    print("RAJNI Diagnostic: FLOPs and Throughput Verification")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    if device != "cuda":
        print("\n⚠️  WARNING: Running on CPU. Timing will be less accurate.")
        print("   For proper benchmarking, use a GPU.")
    
    # Check PyTorch version for torch.compile support
    torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
    has_compile = torch_version >= (2, 0)
    print(f"PyTorch version: {torch.__version__}")
    print(f"torch.compile available: {has_compile}")
    
    # Create model
    print("\n[1] Loading ViT-Base...")
    base_model = timm.create_model('vit_base_patch16_224', pretrained=False)
    base_model.to(device).eval()
    
    # Import RAJNI
    from rajni import AdaptiveJacobianPrunedViT
    from evaluation.flops import flops_reduction
    
    # Wrap with RAJNI
    gamma = 0.05  # Higher gamma for meaningful pruning
    print(f"[2] Creating RAJNI model (gamma={gamma})...")
    rajni_model = AdaptiveJacobianPrunedViT(base_model, gamma=gamma, collect_stats=True)
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
    
    # ============================================
    # PART C: Throughput measurement
    # ============================================
    print("\n" + "=" * 60)
    print("PART C: Throughput Measurement")
    print("=" * 60)
    
    num_iterations = 50
    warmup_iters = 20
    
    # FRESH baseline model
    print("\n[!] Creating FRESH baseline model...")
    baseline_fresh = timm.create_model('vit_base_patch16_224', pretrained=False)
    baseline_fresh.to(device).eval()
    
    # Also create a RAJNI model without stats collection (faster)
    rajni_fast = AdaptiveJacobianPrunedViT(
        timm.create_model('vit_base_patch16_224', pretrained=False).to(device),
        gamma=gamma, 
        collect_stats=False  # Skip stats for speed
    )
    rajni_fast.to(device).eval()
    
    print(f"\nTiming BASELINE ({num_iterations} iterations)...")
    baseline_time = time_model(baseline_fresh, x, num_iterations, warmup_iters, device)
    baseline_throughput = (num_iterations * batch_size) / baseline_time
    
    print(f"Timing RAJNI ({num_iterations} iterations)...")
    rajni_time = time_model(rajni_fast, x, num_iterations, warmup_iters, device)
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
    # PART D: torch.compile test (if available)
    # ============================================
    if has_compile and device == "cuda":
        print("\n" + "=" * 60)
        print("PART D: torch.compile Speedup")
        print("=" * 60)
        
        print("\nCompiling RAJNI model...")
        try:
            rajni_compiled = torch.compile(rajni_fast, mode="reduce-overhead")
            
            # Warmup compiled model (compilation happens on first runs)
            print("Warming up compiled model (this may take a minute)...")
            with torch.no_grad():
                for _ in range(10):
                    _ = rajni_compiled(x)
            torch.cuda.synchronize()
            
            print(f"Timing RAJNI+compile ({num_iterations} iterations)...")
            compiled_time = time_model(rajni_compiled, x, num_iterations, warmup_iters, device)
            compiled_throughput = (num_iterations * batch_size) / compiled_time
            
            compile_speedup = compiled_throughput / baseline_throughput
            
            print(f"\n┌────────────────────────────────────────┐")
            print(f"│  WITH torch.compile                    │")
            print(f"├────────────────────────────────────────┤")
            print(f"│  Compiled: {compiled_throughput:7.1f} img/s ({compiled_time:.2f}s)    │")
            print(f"│  Speedup:  {compile_speedup:7.2f}x (vs baseline)       │")
            print(f"│  vs uncompiled: {compiled_throughput/rajni_throughput:.2f}x           │")
            print(f"└────────────────────────────────────────┘")
        except Exception as e:
            print(f"torch.compile failed: {e}")
            print("This is often due to dynamic control flow. Consider using 'fullgraph=False'")
    
    # ============================================
    # PART E: Consistency Analysis
    # ============================================
    print("\n" + "=" * 60)
    print("PART E: Consistency Analysis")
    print("=" * 60)
    
    flops_speedup = 1.0 / (1.0 - flops['reduction_percent'] / 100)
    
    print(f"\nFLOPs reduction:      {flops['reduction_percent']:.1f}%")
    print(f"Theoretical speedup:  {flops_speedup:.2f}x (from FLOPs)")
    print(f"Measured speedup:     {speedup:.2f}x")
    
    ratio = speedup / flops_speedup if flops_speedup > 1 else speedup
    
    print(f"\nAnalysis:")
    if ratio > 2.0:
        print("  ❌ ERROR: Measured speedup >> theoretical!")
        print("     Check for precision differences (FP16 vs FP32)")
    elif ratio > 1.3:
        print("  ⚠️  Speedup slightly > theoretical")
        print("     May indicate memory/cache benefits from fewer tokens")
    elif speedup < 0.95:
        print("  ⚠️  RAJNI is SLOWER than baseline!")
        print("     Pruning overhead exceeds savings at this gamma.")
        print("     Try increasing gamma for more aggressive pruning.")
    elif ratio < 0.6:
        print("  ⚠️  Speedup < theoretical")
        print("     Normal due to pruning overhead and memory patterns.")
    else:
        print("  ✓ Speedup is consistent with FLOPs reduction!")
    
    print("\n" + "=" * 60)
    print("Diagnostic complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
