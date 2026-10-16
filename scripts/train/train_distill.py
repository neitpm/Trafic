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
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training: Teacher (YOLO11n Base) -> Student (UIB-Nano)")
    parser.add_argument("--teacher-weights", type=str, default="runs/controlled/baseline_seed42/weights/best.pt",
                        help="Path to trained Teacher checkpoint")
    parser.add_argument("--student-model", type=str, default="configs/models/yolo11n_uib_nano.yaml",
                        help="YAML architecture of Student model")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--dis", type=float, default=1.0, help="Distillation loss weight (dis, default=1.0 to balance with YOLO loss)")
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

    register_uib_modules()

    data_yaml = PROJECT_ROOT / "configs" / "dataset" / "vrtsd.yaml"
    teacher_path = PROJECT_ROOT / args.teacher_weights
    student_yaml = PROJECT_ROOT / args.student_model
    output_dir = PROJECT_ROOT / "runs" / "controlled"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"uib_nano_kd_seed{args.seed}"

    print("=" * 65)
    print("KNOWLEDGE DISTILLATION (KD) TRAINING PIPELINE")
    print("=" * 65)
    print(f"Teacher Model: {teacher_path}")
    print(f"Student Model: {student_yaml}")
    print(f"Dataset YAML:  {data_yaml}")
    print(f"Epochs:        {args.epochs}")
    print(f"Image Size:    {args.imgsz}")
    print(f"Batch Size:    {args.batch}")
    print(f"Seed:          {args.seed}")
    print(f"Distill Weight:{args.dis}")
    print(f"Device:        GPU (0) - {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher weights not found at: {teacher_path}")

    # Initialize Student Model from scratch
    student = YOLO(str(student_yaml))

    # Strict Matched Hyperparameters + Distillation Argument
    results = student.train(
        data=str(data_yaml),
        distill_model=str(teacher_path),
        dis=args.dis,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,

        # Optimizer & Schedule (Matched identically with Baseline & Scratch)
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        momentum=0.937,
        warmup_epochs=3.0,
        cos_lr=True,

        # Traffic-Sign Specific Augmentations
        hsv_h=0.012,
        hsv_s=0.65,
        hsv_v=0.35,
        degrees=5.0,
        translate=0.1,
        scale=0.4,
        shear=2.0,
        perspective=0.0005,
        fliplr=0.0,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.05,
        close_mosaic=20,

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

    print(f"\n[OK] Distillation Training completed for UIB-Nano (run: {run_name})")
    print(results)


if __name__ == "__main__":
    main()
