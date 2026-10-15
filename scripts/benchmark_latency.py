import os
import sys
import time
import json
import platform
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Register UIB classes into namespace
import __main__
from app import UIB, UIBDown, ConvBNAct, SqueezeExcite, register_uib_modules

__main__.UIB = UIB
__main__.UIBDown = UIBDown
__main__.ConvBNAct = ConvBNAct
__main__.SqueezeExcite = SqueezeExcite
register_uib_modules()

from ultralytics import YOLO


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_environment_info():
    info = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info.update({
            "device_name": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        })
    else:
        info["device_name"] = "CPU"
    return info


def measure_raw_forward(model, input_tensor, warm_up=50, iters=300, is_cuda=True):
    model.eval()
    with torch.no_grad():
        for _ in range(warm_up):
            _ = model(input_tensor)
        if is_cuda:
            torch.cuda.synchronize()

        latencies = []
        for _ in range(iters):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_tensor)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies)
    return {
        "mean_ms": round(float(np.mean(latencies)), 2),
        "std_ms": round(float(np.std(latencies)), 2),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "fps": round(1000.0 / float(np.mean(latencies)), 1),
    }


def measure_end_to_end_predict(yolo_model, dummy_img_bgr, imgsz=640, warm_up=30, iters=150, device="cuda"):
    """Measures full pipeline: preprocess + raw forward + NMS postprocess via Ultralytics predict."""
    is_cuda = (device == "cuda")
    # Warmup
    for _ in range(warm_up):
        _ = yolo_model.predict(source=dummy_img_bgr, imgsz=imgsz, device=device, verbose=False)
    if is_cuda:
        torch.cuda.synchronize()

    latencies = []
    for _ in range(iters):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = yolo_model.predict(source=dummy_img_bgr, imgsz=imgsz, device=device, verbose=False)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies)
    return {
        "mean_ms": round(float(np.mean(latencies)), 2),
        "std_ms": round(float(np.std(latencies)), 2),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "fps": round(1000.0 / float(np.mean(latencies)), 1),
    }


def main():
    torch.set_num_threads(4)  # Fix CPU threads for deterministic repeatability

    uib_weights = PROJECT_ROOT / "pretrain_backbone_yolo11n" / "train_with_colab" / "vrtsd_uib_p2_fix2" / "weights" / "best.pt"
    baseline_weights = PROJECT_ROOT / "yolo11n" / "runs" / "detect" / "train" / "weights" / "best.pt"

    env_info = get_environment_info()
    print("=" * 65)
    print("ENVIRONMENT & HARDWARE SPECIFICATIONS")
    print("=" * 65)
    for k, v in env_info.items():
        print(f"  {k}: {v}")

    models = [
        ("YOLO11n Baseline", baseline_weights),
        ("YOLO11n-UIB (Proposed)", uib_weights)
    ]

    benchmark_data = {
        "environment": env_info,
        "models": {}
    }

    dummy_img_640 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    dummy_img_832 = np.random.randint(0, 255, (832, 832, 3), dtype=np.uint8)

    for name, path in models:
        print("\n" + "=" * 65)
        print(f"BENCHMARKING: {name}")
        print("=" * 65)
        print(f"Weights file: {path.name} ({path.stat().st_size / (1024*1024):.2f} MB)")

        yolo = YOLO(str(path))
        m = yolo.model
        total_p, trainable_p = count_parameters(m)
        print(f"Parameters: {total_p:,} ({total_p / 1e6:.2f}M)")

        entry = {
            "weights_size_mb": round(path.stat().st_size / (1024*1024), 2),
            "total_params": total_p,
            "params_m": round(total_p / 1e6, 2),
            "raw_forward": {},
            "end_to_end_predict": {}
        }

        # 1. RAW FORWARD PASS BENCHMARK (Pure PyTorch Module)
        print("\n--- 1. Raw Model Forward (Pure PyTorch Module) ---")
        if env_info["cuda_available"]:
            m.cuda()
            # GPU 640x640
            dummy_gpu_640 = torch.randn(1, 3, 640, 640, device="cuda")
            res_gpu_640 = measure_raw_forward(m, dummy_gpu_640, warm_up=50, iters=300, is_cuda=True)
            print(f"[GPU | 640x640] Mean: {res_gpu_640['mean_ms']} ± {res_gpu_640['std_ms']} ms | P50: {res_gpu_640['p50_ms']} ms | P95: {res_gpu_640['p95_ms']} ms | FPS: {res_gpu_640['fps']}")
            entry["raw_forward"]["gpu_640"] = res_gpu_640

            # GPU 832x832
            dummy_gpu_832 = torch.randn(1, 3, 832, 832, device="cuda")
            res_gpu_832 = measure_raw_forward(m, dummy_gpu_832, warm_up=30, iters=200, is_cuda=True)
            print(f"[GPU | 832x832] Mean: {res_gpu_832['mean_ms']} ± {res_gpu_832['std_ms']} ms | P50: {res_gpu_832['p50_ms']} ms | P95: {res_gpu_832['p95_ms']} ms | FPS: {res_gpu_832['fps']}")
            entry["raw_forward"]["gpu_832"] = res_gpu_832

        # CPU 640x640 (Pure forward)
        m.cpu()
        dummy_cpu_640 = torch.randn(1, 3, 640, 640, device="cpu")
        res_cpu_640 = measure_raw_forward(m, dummy_cpu_640, warm_up=15, iters=100, is_cuda=False)
        print(f"[CPU | 640x640] Mean: {res_cpu_640['mean_ms']} ± {res_cpu_640['std_ms']} ms | P50: {res_cpu_640['p50_ms']} ms | P95: {res_cpu_640['p95_ms']} ms | FPS: {res_cpu_640['fps']}")
        entry["raw_forward"]["cpu_640"] = res_cpu_640

        # 2. END-TO-END PIPELINE BENCHMARK (Preprocess + Forward + NMS via Ultralytics)
        print("\n--- 2. End-to-End Pipeline (Preprocess + Forward + NMS) ---")
        if env_info["cuda_available"]:
            e2e_gpu_640 = measure_end_to_end_predict(yolo, dummy_img_640, imgsz=640, warm_up=20, iters=150, device="cuda")
            print(f"[GPU E2E | 640x640] Mean: {e2e_gpu_640['mean_ms']} ± {e2e_gpu_640['std_ms']} ms | P50: {e2e_gpu_640['p50_ms']} ms | P95: {e2e_gpu_640['p95_ms']} ms | FPS: {e2e_gpu_640['fps']}")
            entry["end_to_end_predict"]["gpu_640"] = e2e_gpu_640

            e2e_gpu_832 = measure_end_to_end_predict(yolo, dummy_img_832, imgsz=832, warm_up=20, iters=100, device="cuda")
            print(f"[GPU E2E | 832x832] Mean: {e2e_gpu_832['mean_ms']} ± {e2e_gpu_832['std_ms']} ms | P50: {e2e_gpu_832['p50_ms']} ms | P95: {e2e_gpu_832['p95_ms']} ms | FPS: {e2e_gpu_832['fps']}")
            entry["end_to_end_predict"]["gpu_832"] = e2e_gpu_832

        e2e_cpu_640 = measure_end_to_end_predict(yolo, dummy_img_640, imgsz=640, warm_up=5, iters=30, device="cpu")
        print(f"[CPU E2E | 640x640] Mean: {e2e_cpu_640['mean_ms']} ± {e2e_cpu_640['std_ms']} ms | P50: {e2e_cpu_640['p50_ms']} ms | P95: {e2e_cpu_640['p95_ms']} ms | FPS: {e2e_cpu_640['fps']}")
        entry["end_to_end_predict"]["cpu_640"] = e2e_cpu_640

        benchmark_data["models"][name] = entry

    out_file = PROJECT_ROOT / "benchmark_latency_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\n[OK] Complete profiling results with metadata saved to: {out_file}")


if __name__ == "__main__":
    main()
