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
                        help="Path to JSON file containing RAJNI pruning schedule")

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
    if args.schedule is None:
        raise ValueError("You must provide --schedule for RAJNI evaluation")

    with open(args.schedule, "r") as f:
        pruning_schedule = json.load(f)

    print("\nLoaded RAJNI schedule:")
    for k, v in pruning_schedule.items():
        print(f"  Layer {k}: {v}")

    rajni_model = RAJNIViTWrapper(
        base_model=timm.create_model(
            args.model,
            pretrained=True,
        ),
        pruning_schedule=pruning_schedule,
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