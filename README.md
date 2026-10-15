# Trafic: Vietnamese Traffic Sign Detection with Custom YOLO11n-UIB

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3+-00D2FF.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical research and engineering prototype evaluating **Universal Inverted Bottleneck (UIB)** backbone integration into the **Ultralytics YOLO11** framework for Vietnamese traffic sign localization and classification (VRTSD - 58 categories), paired with an interactive real-time inference web application.

---

## 1. Project Overview & Motivation

Detecting traffic signs in Vietnam presents distinct real-world challenges:
- High density of diverse regulatory, warning, and prohibitory signage (QCVN 41:2019/BGTVT standards).
- Complex urban background clutter, variable illumination, and small object scales relative to total canvas resolution.

This repository explores modifying the standard **YOLO11n** architecture by replacing default C3k2/convolutional stages with custom **Universal Inverted Bottleneck (UIB)** blocks (featuring depthwise separable convolutions, inverted bottleneck expansion, and Squeeze-and-Excitation attention).

---

## 2. Model Architectures & Parameter Profiling

The repository evaluates two distinct model configurations (parameters verified directly from fused model checkpoints via `scripts/benchmark_latency.py`):

| Architectural Property | Baseline (YOLO11n) | Proposed Variant (YOLO11n-UIB-P2) | Structural Notes |
| :--- | :---: | :---: | :--- |
| **Backbone Stages** | Standard CSPDarknet (Conv + C3k2) | **Custom UIB & UIBDown Blocks** | Depthwise conv + Expansion ($e=6.0$) + Squeeze-and-Excitation |
| **Detection Heads** | 3 scales (P3, P4, P5) | **4 scales (P2, P3, P4, P5)** | Added high-resolution P2 head for small traffic sign localization |
| **Fused Model Parameters** | **2,593,657 (~2.59M)** | **31,948,198 (~31.95M)** | Higher capacity variant (~12.3x parameter expansion) |
| **Fused Layers** | **100 layers** | **348 layers** | Deep inverted bottleneck topology |
| **GFLOPs (fused @ 640x640)** | **6.4 GFLOPs** | **52.9 GFLOPs** | ~8.2x computational workload |
| **Weights File Size** | **5.25 MB** | **61.96 MB** | PyTorch serialized checkpoint |

> **Important Architectural Note:**  
> While the baseline model operates as a true lightweight detector (~2.59M parameters), the custom UIB variant was designed with wider channel dimensions (64 -> 128 -> 256 -> 512) and an additional P2 high-resolution detection head. Consequently, the UIB variant functions as a **higher-capacity, accuracy-oriented model** rather than an edge-nano detector.

---

## 3. Rigorous Profiling & Latency Benchmark

To ensure scientific honesty and consistency between reported documentation and artifacts, profiling is generated directly by `scripts/benchmark_latency.py` and saved to `benchmark_latency_results.json`. Both models were evaluated as fused modules (`model.fuse()`), using `torch.set_num_threads(4)` on CPU and hardware-synchronized timing via `torch.cuda.Event` on GPU.

- **Benchmark Environment:**
  - **OS:** Windows 11 (10.0.26200)
  - **GPU:** NVIDIA GeForce RTX 5060 Ti (15.93 GB VRAM)
  - **CUDA:** 12.8 | **cuDNN:** 92000
  - **PyTorch:** 2.12.0.dev+cu128 | **Ultralytics:** 8.4.177
  - **Environment Pinning:** See [`requirements-lock.txt`](requirements-lock.txt)

### Model Inference Latency (Batch Size = 1, Fused Models, Synchronized)

| Target Hardware | Resolution | Metric | Baseline (`yolo11n_baseline_best.pt`) | UIB Variant (`yolo11n_uib_best.pt`) |
| :--- | :---: | :--- | :---: | :---: |
| **GPU RTX 5060 Ti** | **640x640** | **Mean ± Std**<br>Median (p50) / p95<br>**Throughput (FPS)** | **5.21 ± 0.72 ms**<br>4.99 ms / 6.54 ms<br>**192.0 FPS** | **16.43 ± 1.03 ms**<br>16.10 ms / 18.84 ms<br>**60.9 FPS** |
| **GPU RTX 5060 Ti** | **832x832** | **Mean ± Std**<br>Median (p50) / p95<br>**Throughput (FPS)** | **5.19 ± 0.67 ms**<br>4.98 ms / 6.72 ms<br>**192.8 FPS** *(GPU underutilized)* | **26.26 ± 0.27 ms**<br>26.20 ms / 26.72 ms<br>**38.1 FPS** |
| **CPU (4 threads)** | **640x640** | **Mean ± Std**<br>Median (p50) / p95<br>**Throughput (FPS)** | **30.29 ± 2.57 ms**<br>29.43 ms / 35.32 ms<br>**33.0 FPS** | **468.10 ± 85.15 ms**<br>426.12 ms / 633.64 ms<br>**2.1 FPS** |

*Hardware Profiling Observations:*
1. **GPU Throughput:** On an RTX 5060 Ti workstation, the higher-capacity UIB variant comfortably sustains **60.9 FPS** at 640x640 and **38.1 FPS** at 832x832, suitable for real-time video stream ingestion.
2. **CPU Scalability:** On CPU inference, the baseline model achieves **33.0 FPS**, while the 32M-parameter UIB model requires **~468 ms per frame (2.1 FPS)**, confirming the strict computational cost of the expanded backbone.

---

## 4. Detection Accuracy & Experimental Artifacts

To maintain methodological consistency, all reported validation metrics correspond to the **Best mAP@[0.5:0.95] Checkpoint Policy** derived from official training logs on the Vietnamese Real-world Traffic Sign Dataset (VRTSD, 58 classes):

| Metric | Baseline YOLO11n (best.pt) | YOLO11n-UIB (best.pt) | Metric Selection Policy |
| :--- | :---: | :---: | :--- |
| **Optimal Epoch** | **Epoch 85** | **Epoch 151** | Selected strictly by peak mAP50-95 |
| **mAP@[0.5:0.95] (Peak)** | **0.6970 (69.7%)** | **0.7131 (71.3%)** | Primary ranking criteria |
| **mAP@0.5** | **0.8488 (84.9%)** | **0.8848 (88.5%)** | Evaluated at the optimal epoch |
| **Precision** | **0.8761 (87.6%)** | **0.9194 (91.9%)** | Evaluated at the optimal epoch |
| **Recall** | **0.7822 (78.2%)** | **0.8453 (84.5%)** | Evaluated at the optimal epoch |
| **Training Budget** | 100 epochs, imgsz=640, bs=16, SGD/auto | 200 epochs, imgsz=832, bs=8, AdamW | Confounding recipe differences |

### Checkpoint Provenance & Artifact Integrity

| Checkpoint File | Fused Parameters | Size | SHA-256 Checksum | Source Log |
| :--- | :---: | :---: | :---: | :--- |
| `weights/yolo11n_baseline_best.pt` | 2,593,657 | 5.25 MB | `bfce0abf23c38e3274e50779f9083b5d0d5cdd4c5070beaa55fa97a9f3df703e` | yolo11n/runs/detect/train/results.csv |
| `weights/yolo11n_uib_best.pt` | 31,948,198 | 61.96 MB | `bca8ac397d978dbd30ef4430b5970f5e52643c241ba1edcb0f514fb4dcea4c8f` | pretrain_backbone_yolo11n/train_with_colab/vrtsd_uib_p2_fix2/results.csv |
> **Methodological & Reproducibility Disclaimer:**  
> The higher mAP observed in the UIB variant was achieved under an extended training budget (200 epochs vs. 100 epochs) and higher input resolution (832x832 vs. 640x640) with ~12x model capacity. While these metrics confirm the strong detection capability of the trained checkpoint, isolating the exact percentage gain attributable strictly to the UIB architecture requires a controlled ablation under identical training recipes and seeds.

---

## 5. Repository Structure

```bash
Trafic/
├── configs/
│   ├── dataset/
│   │   └── vrtsd.yaml             # Dataset configuration (58 categories)
│   └── models/
│       └── yolo11n_uib_backbone.yaml # Custom UIB architecture definition
├── weights/
│   ├── yolo11n_baseline_best.pt   # Serialized baseline checkpoint (2.59M params, 5.25 MB)
│   └── yolo11n_uib_best.pt        # Serialized UIB variant checkpoint (31.95M params, 61.96 MB)
├── scripts/
│   ├── benchmark_latency.py       # Hardware-synchronized latency and FPS profiling
│   ├── eval/                      # Validation and test evaluation scripts
│   ├── train/                     # Training entrypoints
│   └── infer/                     # Batch prediction scripts
├── src/
│   ├── models/                    # Modular UIB implementation and Ultralytics registry
│   └── utils/                     # Metrics and plotting helpers
├── tests/
│   └── test_uib.py                # Unit tests for UIB tensor math and checkpoint loading
├── app.py                         # Streamlit real-time interactive demo web app
├── requirements.txt               # Main dependencies
├── requirements-lock.txt          # Pinned exact environment freeze
└── LICENSE                        # MIT License
```

---

## 6. Installation & Quickstart

### 1. Clone & Setup Environment

```bash
git clone https://github.com/neitpm/Trafic.git
cd Trafic

# Recommended: Python 3.10+
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
# To reproduce the exact profiling environment:
# pip install -r requirements-lock.txt
```

### 2. Run Technical Unit Tests

Verify UIB residual logic, stride downsampling, checkpoint loading, and end-to-end NMS predict pipeline:

```bash
python -m unittest tests/test_uib.py
```

### 3. Run Hardware Latency Benchmark

Profile raw forward pass and throughput on your local machine:

```bash
python scripts/benchmark_latency.py
```

### 4. Launch Interactive Streamlit Demo

Launch the traffic sign detection web app supporting Images, MP4 Videos, and Live Webcam streaming:

```bash
streamlit run app.py
```

---

## 7. Citation & License

This project is licensed under the [MIT License](LICENSE).
Trained weights and architectures are intended for academic research, benchmarking, and prototyping.