import argparse
import os
import json
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import timm

from .eval import evaluate_model
from .wrapper import RAJNIViTWrapper   # adjust filename if needed


# --------------------------------------------------
# Args
# --------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser("RAJNI Evaluation", add_help=True)

    # Dataset / loader
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to ImageNet-style dataset root")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_mem", action="store_true", default=True)

    # Model
    parser.add_argument("--model", type=str, default="vit_base_patch16_224",
                        help="timm model name")
    parser.add_argument("--device", type=str, default="cuda")

    # RAJNI
    parser.add_argument("--schedule", type=str, default=None,
                        help="Path to JSON file containing RAJNI dynamic parameters (optional)")
    parser.add_argument("--percentile", type=float, default=0.75,
                        help="Percentile for q-norm difficulty computation")
    parser.add_argument("--kr_min", type=float, default=0.60,
                        help="Minimum keep ratio")
    parser.add_argument("--gamma", type=float, default=2.5,
                        help="Exponential decay rate for keep ratio")
    parser.add_argument("--skip_layers", type=int, nargs="+", default=[10, 11],
                        help="Layer indices to skip pruning (full attention)")

    # Eval
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit number of batches for fast eval")
    parser.add_argument("--compare_base", action="store_true",
                        help="Compare with base (unpruned) model")

    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = get_args()

    device = torch.device(args.device)
    cudnn.benchmark = True

    print("\nArgs:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_dir = args.data_path
    dataset_val = datasets.ImageFolder(val_dir, transform=transform_val)

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    print(f"\nLoaded validation set: {len(dataset_val)} images")

    # --------------------------------------------------
    # Base model
    # --------------------------------------------------
    base_model = timm.create_model(
        args.model,
        pretrained=True,
    ).to(device).eval()

    # --------------------------------------------------
    # Optional: base benchmark
    # --------------------------------------------------
    if args.compare_base:
        print("\n🔹 Evaluating BASE model")
        base_acc, base_throughput = evaluate_model(
            base_model,
            val_loader,
            device=device,
            warmup=args.warmup,
            max_batches=args.max_batches,
        )

        print(
            f"Base  - Accuracy: {base_acc:.2f}%, "
            f"Throughput: {base_throughput:.1f} img/s"
        )

    # --------------------------------------------------
    # RAJNI model
    # --------------------------------------------------
    # Load dynamic parameters from schedule file if provided
    if args.schedule is not None:
        with open(args.schedule, "r") as f:
            dynamic_params = json.load(f)
        
        percentile = dynamic_params.get("percentile", args.percentile)
        kr_min = dynamic_params.get("kr_min", args.kr_min)
        gamma = dynamic_params.get("gamma", args.gamma)
        skip_layers = tuple(dynamic_params.get("skip_layers", args.skip_layers))
        
        print("\nLoaded RAJNI dynamic parameters:")
        print(f"  percentile: {percentile}")
        print(f"  kr_min: {kr_min}")
        print(f"  gamma: {gamma}")
        print(f"  skip_layers: {skip_layers}")
    else:
        percentile = args.percentile
        kr_min = args.kr_min
        gamma = args.gamma
        skip_layers = tuple(args.skip_layers)
        
        print("\nUsing default RAJNI dynamic parameters:")
        print(f"  percentile: {percentile}")
        print(f"  kr_min: {kr_min}")
        print(f"  gamma: {gamma}")
        print(f"  skip_layers: {skip_layers}")

    rajni_model = RAJNIViTWrapper(
        base_model=timm.create_model(
            args.model,
            pretrained=True,
        ),
        percentile=percentile,
        kr_min=kr_min,
        gamma=gamma,
        skip_layers=skip_layers,
    ).to(device).eval()

    print("\n🔹 Evaluating RAJNI model")
    rajni_acc, rajni_throughput = evaluate_model(
        rajni_model,
        val_loader,
        device=device,
        warmup=args.warmup,
        max_batches=args.max_batches,
    )

    print(
        f"RAJNI - Accuracy: {rajni_acc:.2f}%, "
        f"Throughput: {rajni_throughput:.1f} img/s"
    )

    # --------------------------------------------------
    # Speedup
    # --------------------------------------------------
    if args.compare_base:
        speedup = rajni_throughput / base_throughput
        acc_drop = base_acc - rajni_acc

        print(
            f"\n🚀 Speedup: {speedup:.2f}x | "
            f"Accuracy drop: {acc_drop:.2f}%"
        )


if __name__ == "__main__":
    main()