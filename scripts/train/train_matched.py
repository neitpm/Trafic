import argparse
import os
import sys
import gc
import random
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from ultralytics import YOLO

from src.models.register import register_uib_modules


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Matched Recipe Training for VR-TSD Traffic Signs")
    parser.add_argument("--model-type", type=str, choices=["baseline", "uib", "uib_nano"], required=True,
                        help="Model architecture: 'baseline', 'uib' (32M), or 'uib_nano' (2.48M)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (fits RTX 5060 Ti 16GB comfortably)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Deterministic Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
    gc.collect()

    data_yaml = PROJECT_ROOT / "configs" / "dataset" / "vrtsd.yaml"
    output_dir = PROJECT_ROOT / "runs" / "controlled"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("CONTROLLED BENCHMARK TRAINING PIPELINE")
    print("=" * 65)
    print(f"Model Type:   {args.model_type.upper()}")
    print(f"Dataset YAML: {data_yaml}")
    print(f"Epochs:       {args.epochs}")
    print(f"Image Size:   {args.imgsz}")
    print(f"Batch Size:   {args.batch}")
    print(f"Seed:         {args.seed}")
    print(f"Device:       GPU (0) - {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    if args.model_type == "baseline":
        # Baseline uses standard pretrained YOLO11n checkpoint
        pretrained_weights = PROJECT_ROOT / "weights" / "yolo11n.pt"
        if not pretrained_weights.exists():
            pretrained_weights = PROJECT_ROOT / "yolo11n.pt"
        model = YOLO(str(pretrained_weights))
        run_name = f"baseline_seed{args.seed}"
    elif args.model_type == "uib_nano":
        register_uib_modules()
        model_yaml = PROJECT_ROOT / "configs" / "models" / "yolo11n_uib_nano.yaml"
        model = YOLO(str(model_yaml))
        run_name = f"uib_nano_seed{args.seed}"
    else:
        # Custom UIB backbone with registered modules
        register_uib_modules()
        model_yaml = PROJECT_ROOT / "configs" / "models" / "yolo11n_uib_backbone.yaml"
        pretrained_weights = PROJECT_ROOT / "weights" / "yolo11n.pt"
        if not pretrained_weights.exists():
            pretrained_weights = PROJECT_ROOT / "yolo11n.pt"

        model = YOLO(str(model_yaml))
        model.load(str(pretrained_weights))
        run_name = f"uib_seed{args.seed}"

    # Unified Training Recipe (Identical across both baseline and UIB)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,

        # Optimizer & Schedule
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        momentum=0.937,
        warmup_epochs=3.0,
        cos_lr=True,

        # Traffic-Sign Specific Augmentations (Crucial: fliplr=0.0 to preserve direction labels)
        hsv_h=0.012,
        hsv_s=0.65,
        hsv_v=0.35,
        degrees=5.0,
        translate=0.1,
        scale=0.4,
        shear=2.0,
        perspective=0.0005,
        fliplr=0.0,       # No horizontal flipping for traffic signs
        flipud=0.0,
        mosaic=1.0,
        mixup=0.05,
        close_mosaic=20,  # Turn off mosaic last 20 epochs for fine-grained convergence

        # Logging & Validation
        val=True,
        save=True,
        plots=True,
        patience=50,
        save_period=10,
        project=str(output_dir),
        name=run_name,
        exist_ok=True,
        seed=args.seed,
    )

    print(f"\n[OK] Training completed for {args.model_type} (run: {run_name})")
    print(results)


if __name__ == "__main__":
    main()
