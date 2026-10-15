import os
import sys
import time
import json
import platform
import numpy as np
import torch
import cv2
from pathlib import Path

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


def profile_model(name, weights_path, is_cuda=True):
    print(f"\n=======================================================")
    print(f"PROFILING: {name}")
    print(f"=======================================================")
    weights_path = Path(weights_path)
    file_size_mb = round(weights_path.stat().st_size / (1024 * 1024), 2)

    yolo = YOLO(str(weights_path))
    model = yolo.model.eval()

    # Fuse Conv+BatchNorm exactly as Ultralytics does for production inference
    try:
        model.fuse()
    except Exception as e:
        print(f"Note on fuse: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters (fused): {total_params:,} ({total_params / 1e6:.2f}M) | File: {file_size_mb} MB")

    res = {
        "weights_file": weights_path.name,
        "weights_size_mb": file_size_mb,
        "parameters_fused": total_params,
        "params_m": round(total_params / 1e6, 2),
        "gpu_inference": {},
        "cpu_inference": {}
    }

    # 1. GPU BENCHMARK (torch.cuda.Event, fused model)
    if is_cuda:
        model.cuda()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for sz in [640, 832]:
            x = torch.randn(1, 3, sz, sz, device="cuda")
            # Warmup
            for _ in range(50):
                _ = model(x)
            torch.cuda.synchronize()

            times = []
            for _ in range(300):
                starter.record()
                _ = model(x)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))

            times = np.array(times)
            res["gpu_inference"][f"imgsz_{sz}"] = {
                "mean_ms": round(float(np.mean(times)), 2),
                "std_ms": round(float(np.std(times)), 2),
                "p50_ms": round(float(np.percentile(times, 50)), 2),
                "p95_ms": round(float(np.percentile(times, 95)), 2),
                "p99_ms": round(float(np.percentile(times, 99)), 2),
                "fps": round(1000.0 / float(np.mean(times)), 1)
            }
            print(f"[GPU {sz}x{sz}] Mean: {res['gpu_inference'][f'imgsz_{sz}']['mean_ms']} ± {res['gpu_inference'][f'imgsz_{sz}']['std_ms']} ms | P50: {res['gpu_inference'][f'imgsz_{sz}']['p50_ms']} ms | P95: {res['gpu_inference'][f'imgsz_{sz}']['p95_ms']} ms | FPS: {res['gpu_inference'][f'imgsz_{sz}']['fps']}")

    # 2. CPU BENCHMARK (Same fused instance, 4 threads, time.perf_counter)
    model.cpu()
    torch.set_num_threads(4)
    x_cpu = torch.randn(1, 3, 640, 640, device="cpu")
    # Warmup
    for _ in range(15):
        _ = model(x_cpu)

    cpu_times = []
    # 80 iterations for statistically robust CPU measurement
    iters = 80 if total_params > 10_000_000 else 150
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model(x_cpu)
        t1 = time.perf_counter()
        cpu_times.append((t1 - t0) * 1000.0)

    cpu_times = np.array(cpu_times)
    res["cpu_inference"]["imgsz_640"] = {
        "mean_ms": round(float(np.mean(cpu_times)), 2),
        "std_ms": round(float(np.std(cpu_times)), 2),
        "p50_ms": round(float(np.percentile(cpu_times, 50)), 2),
        "p95_ms": round(float(np.percentile(cpu_times, 95)), 2),
        "p99_ms": round(float(np.percentile(cpu_times, 99)), 2),
        "fps": round(1000.0 / float(np.mean(cpu_times)), 1)
    }
    print(f"[CPU 640x640] Mean: {res['cpu_inference']['imgsz_640']['mean_ms']} ± {res['cpu_inference']['imgsz_640']['std_ms']} ms | P50: {res['cpu_inference']['imgsz_640']['p50_ms']} ms | P95: {res['cpu_inference']['imgsz_640']['p95_ms']} ms | FPS: {res['cpu_inference']['imgsz_640']['fps']}")

    return res


def main():
    uib_weights = PROJECT_ROOT / "weights" / "yolo11n_uib_best.pt"
    baseline_weights = PROJECT_ROOT / "weights" / "yolo11n_baseline_best.pt"

    env = get_environment_info()
    is_cuda = env["cuda_available"]

    data = {
        "environment": env,
        "models": {
            "YOLO11n Baseline": profile_model("YOLO11n Baseline", baseline_weights, is_cuda=is_cuda),
            "YOLO11n-UIB (Proposed)": profile_model("YOLO11n-UIB (Proposed)", uib_weights, is_cuda=is_cuda)
        }
    }

    out_file = PROJECT_ROOT / "benchmark_latency_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n[OK] Latency results cleanly generated and saved to: {out_file}")


if __name__ == "__main__":
    main()
