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

The repository evaluates two distinct model configurations:

| Architectural Property | Baseline (YOLO11n) | Proposed Variant (YOLO11n-UIB-P2) | Structural Notes |
| :--- | :---: | :---: | :--- |
| **Backbone Stages** | Standard CSPDarknet (Conv + C3k2) | **Custom UIB & UIBDown Blocks** | Depthwise conv + Expansion ($e=6.0$) + Squeeze-and-Excitation |
| **Detection Heads** | 3 scales (P3, P4, P5) | **4 scales (P2, P3, P4, P5)** | Added high-resolution P2 head for small traffic sign localization |
| **Total Parameters** | **2,593,657 (~2.59M)** | **31,948,198 (~31.95M)** | Higher capacity variant (~12.3x parameter expansion) |
| **GFLOPs (fused @ 640x640)** | **6.4 GFLOPs** | **52.9 GFLOPs** | ~8.2x computational workload |
| **Weights File Size** | **5.25 MB** | **61.96 MB** | PyTorch serialized checkpoint |

> **Important Architectural Note:**  
> While the baseline model operates as a true lightweight detector (~2.6M parameters), the custom UIB variant was designed with wider channel dimensions (64 -> 128 -> 256 -> 512) and an additional P2 high-resolution detection head. Consequently, the UIB variant functions as a **higher-capacity, accuracy-oriented model** rather than an edge-nano detector.

---

## 3. Rigorous Profiling & Latency Benchmark

To ensure scientific honesty and avoid latency misreporting, profiling was conducted under identical hardware conditions with `torch.set_num_threads(4)`, Conv+BatchNorm layer fusion (`model.fuse()`), and hardware-synchronized timing using `torch.cuda.Event`.

- **Benchmark Environment:**
  - **OS:** Windows 11 (10.0.26200)
  - **GPU:** NVIDIA GeForce RTX 5060 Ti (15.93 GB VRAM)
  - **CUDA:** 12.8 | **cuDNN:** 92000
  - **PyTorch:** 2.12.0.dev+cu128 | **Ultralytics:** 8.4.177

### Model Inference Latency (Batch Size = 1, Fused Models)

| Target Hardware | Resolution | Metric | Baseline (2.6M params) | UIB Variant (31.9M params) |
| :--- | :---: | :--- | :---: | :---: |
| **GPU RTX 5060 Ti** | **640x640** | **Mean ± Std**<br>Median (p50) / p95<br>**Throughput (FPS)** | **6.40 ± 1.95 ms**<br>5.75 ms / 9.92 ms<br>**156.2 FPS** | **17.17 ± 0.79 ms**<br>17.10 ms / 18.13 ms<br>**58.2 FPS** |
| **GPU RTX 5060 Ti** | **832x832** | **Mean ± Std**<br>Median (p50) / p95<br>**Throughput (FPS)** | **5.56 ± 0.76 ms**<br>5.40 ms / 6.68 ms<br>**179.9 FPS** *(GPU underutilized)* | **27.48 ± 0.49 ms**<br>27.54 ms / 28.22 ms<br>**36.4 FPS** |
| **CPU (4 threads)** | **640x640** | **Mean ± Std**<br>Median (p50) / p95<br>**Throughput (FPS)** | **46.10 ± 10.02 ms**<br>46.49 ms / 61.20 ms<br>**21.7 FPS** | **405.64 ± 31.69 ms**<br>399.85 ms / 453.89 ms<br>**2.5 FPS** |

*Hardware Profiling Observations:*
1. **GPU Throughput:** On an RTX 5060 Ti, the UIB variant sustains **58.2 FPS** at 640x640 and **36.4 FPS** at 832x832, comfortably meeting real-time camera streaming thresholds for workstation deployment.
2. **CPU Scalability:** On CPU inference, the baseline executes at **21.7 FPS**, whereas the 32M-parameter UIB model requires **~405 ms per frame (2.5 FPS)**, demonstrating the strict trade-off of the added capacity.

---

## 4. Detection Accuracy & Experimental Artifacts

Historical validation results tracked across training logs on the Vietnamese Real-world Traffic Sign Dataset (VRTSD, 58 classes):

| Metric | Baseline YOLO11n | YOLO11n-UIB (Tracked Run) | Evaluation Criteria |
| :--- | :---: | :---: | :--- |
| **Best mAP@0.5** | **0.8389 (83.9%)** | **0.8848 (88.5%)** | Peak mAP50 recorded on validation split |
| **Best mAP@[0.5:0.95]** | **0.6822 (68.2%)** | **0.7131 (71.3%)** | Peak COCO mAP recorded (Epoch 151) |
| **Precision** | **0.8895 (88.9%)** | **0.9241 (92.4%)** | Detection confidence thresholding |
| **Recall** | **0.7888 (78.9%)** | **0.8454 (84.5%)** | Positive detection rate |
| **Training Recipe** | 100 epochs, imgsz=640, bs=16, SGD/auto | 200 epochs, imgsz=832, bs=8, AdamW | Distinct optimization budgets |

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
│   ├── yolo11n_baseline_best.pt   # Serialized baseline checkpoint (2.6M params)
│   └── yolo11n_uib_best.pt        # Serialized UIB variant checkpoint (31.9M params)
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
└── requirements.txt               # Environment dependencies
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
```

### 2. Run Technical Unit Tests

Verify UIB residual logic, stride downsampling, and checkpoint loading:

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