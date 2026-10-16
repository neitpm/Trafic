# Trafic: Vietnamese Traffic Sign Detection with Custom YOLO11n-UIB

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4+-00D2FF.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical research and engineering benchmark evaluating **Universal Inverted Bottleneck (UIB)** architectural integration into the **Ultralytics YOLO11** framework for Vietnamese traffic sign localization and classification (VR-TSD - 58 categories), paired with an interactive real-time inference web application.

---

## 1. Project Overview & Motivation

Detecting traffic signs in Vietnam presents distinct real-world challenges:
- High density of diverse regulatory, warning, and prohibitory signage (QCVN 41:2019/BGTVT standards).
- Complex urban background clutter, variable illumination, and small object scales relative to total canvas resolution.

This repository explores modifying the standard **YOLO11n** architecture by replacing default C3k2/convolutional stages with custom **Universal Inverted Bottleneck (UIB)** blocks (featuring depthwise separable convolutions, inverted bottleneck expansion, and Squeeze-and-Excitation attention).

---

## 2. Experimental Methodology & Rigorous Controlled Benchmark

To eliminate confounding variables (different epochs, resolutions, augmentations, and seeds), all controlled evaluations were conducted under an identical, standardized training protocol:
- **Dataset:** Vietnamese Real-world Traffic Sign Dataset v2 (VR-TSD-2, 58 classes: 5,668 train, 1,259 val, 1,151 test).
- **Standardized Recipe:** `imgsz=640`, `epochs=200`, `batch=8`, `optimizer=AdamW` (`lr0=0.001`, `lrf=0.01`, `weight_decay=0.0005`, `momentum=0.937`), `cos_lr=True`, `warmup_epochs=3.0`, `fliplr=0.0` (preserves directional sign semantics), `close_mosaic=20`, `seed=42`.
- **Hardware:** Workstation equipped with NVIDIA GeForce RTX 5060 Ti (16GB VRAM), CUDA 12.8, PyTorch 2.12.0.dev.
- **Latency Benchmarking:** Measured in fused mode (`model.fuse()`) across 500 iterations using hardware-synchronized `torch.cuda.Event` timing.
- **Checkpoint Policy:** Strictly chosen based on the peak **mAP@[0.5:0.95]** on the validation split.

### Full Experimental Benchmark Results

| Model Configuration | Training Paradigm | Fused Params | FLOPs (640x640) | Mean Latency (RTX 5060 Ti) | Throughput (FPS) | mAP@[0.5:0.95] | mAP@0.5 | Precision | Recall | Val Box Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11n Baseline** | COCO-Pretrained Fine-tune | 2.61M | 6.5 G | **5.79 ms** | **172.8 FPS** | **70.27%** | **87.63%** | **92.19%** | **81.44%** | **0.832** |
| **YOLO11n-UIB (Original)** | Trained from Scratch (P2–P5) | 31.78M | 51.9 G | 15.20 ms | 60.9 FPS | **70.86%** | 86.80% | 89.39% | **82.51%** | **0.787** |
| **UIB-Nano (Scratch)** | Trained from Scratch (P3–P5) | **2.48M** | **5.6 G** | 6.93 ms | 144.4 FPS | 61.20% | **83.71%** | 89.65% | 77.72% | 1.028 |
| **UIB-Nano + KD ($\text{dis}=0.5$)** | Knowledge Distillation | **2.48M** | **5.6 G** | 6.93 ms | 144.4 FPS | 63.59% | 81.85% | 89.71% | 77.09% | 0.922 |
| **UIB-Nano + KD ($\text{dis}=1.0$)** | Knowledge Distillation | **2.48M** | **5.6 G** | 6.93 ms | 144.4 FPS | **64.05%** | 82.52% | 87.87% | 77.06% | **0.917** |
| **UIB-Nano + KD ($\text{dis}=2.0$)** | Knowledge Distillation | **2.48M** | **5.6 G** | 6.93 ms | 144.4 FPS | 62.08% | 79.85% | 82.08% | 75.95% | 0.948 |

> **Key Takeaway:**  
> The experiments quantify the trade-off between model compactness, inference cost, and detection accuracy, while measuring the partial benefit of teacher-student distillation across a pre-registered hyperparameter grid.

---

## 3. Detailed Technical Analysis

### 3.1. UIB-Nano Compactness vs. Baseline
- **Parameter & Computation Efficiency:** UIB-Nano reduces parameter count by **5.14%** (2.48M vs. 2.61M) and computational workload by **13.85%** (5.6 vs. 6.5 GFLOPs) compared to YOLO11n Baseline.
- **Inference Latency Profile:** Despite having fewer FLOPs, UIB-Nano's latency on GPU is approximately **1.20x higher** (6.93 ms vs. 5.79 ms). This latency profile is a well-documented hardware characteristic of Depthwise Separable Convolutions, which incur higher Memory Access Cost (MAC) relative to arithmetic FLOP count. Nonetheless, at **144.4 FPS**, UIB-Nano comfortably exceeds standard real-time requirements (>100 FPS).

### 3.2. Impact of Knowledge Distillation Grid ($\text{dis} \in \{0.5, 1.0, 2.0\}$)
- **Ablation Trends:**
  - The evaluated distillation weights exhibited a non-monotonic response across the tested range ($\text{dis} \in \{0.5, 1.0, 2.0\}$).
  - $\text{dis}=1.0$ achieved the best validation mAP among the tested grid values, peaking at **64.05% mAP@[0.5:0.95]** (+2.85 points over Scratch) and the lowest validation box loss (**0.917**).
  - Lower distillation weight ($\text{dis}=0.5$) relaxed teacher constraints and recovered higher Precision (**89.71%**), but produced a slightly lower mAP@[0.5:0.95] (**63.59%**).
  - Higher distillation weight ($\text{dis}=2.0$) penalized the student's task-specific detection loss, resulting in lower mAP@[0.5:0.95] (**62.08%**) and Precision (**82.08%**).
- **Scope & Limitations:** While KD improved bounding box localization relative to pure scratch training, under this teacher, loss formulation, seed 42, and 200-epoch protocol, KD did not close the remaining 6.22-point gap to the COCO-pretrained Baseline. Pretraining on large-scale datasets remains a dominant factor in this ultra-compact regime.

### 3.3. High-Capacity UIB Variant (31.8M)
- The high-capacity UIB model achieved **70.86% mAP@[0.5:0.95]** and **82.51% Recall** from scratch, demonstrating the representational power of deep UIB blocks when capacity is unconstrained, albeit at ~12.2x higher parameter footprint and ~52 GFLOPs.

---

## 4. Repository Structure

```bash
Trafic/
├── configs/
│   ├── dataset/
│   │   └── vrtsd.yaml                  # Dataset configuration (58 categories)
│   ├── models/
│   │   ├── yolo11n_uib_backbone.yaml   # 31.8M UIB architecture (P2–P5)
│   │   └── yolo11n_uib_nano.yaml       # 2.48M lightweight UIB architecture (P3–P5)
│   └── benchmark_recipe_manifest.json  # Machine-readable experimental manifest
├── runs/controlled/
│   ├── baseline_seed42/                # Standard YOLO11n 200 epochs run
│   ├── uib_seed42/                     # Original UIB 32M 200 epochs run
│   ├── uib_nano_seed42/                # UIB-Nano scratch 200 epochs run
│   └── uib_nano_kd_seed42/             # UIB-Nano + Knowledge Distillation run
├── scripts/
│   ├── train/
│   │   ├── train_matched.py            # Reproducible matched recipe training script
│   │   └── train_distill.py            # Teacher-student Knowledge Distillation pipeline
│   ├── benchmark_latency.py            # Hardware-synchronized GPU/CPU latency profiler
│   ├── eval/                           # Validation and test evaluation scripts
│   └── infer/                          # Batch inference scripts
├── src/
│   ├── models/                         # Modular UIB/UIBDown implementation & registry
│   └── utils/                          # Metric and plotting helpers
├── tests/
│   └── test_uib.py                     # Unit tests for UIB tensor math & graph checks
├── app.py                              # Streamlit real-time interactive demo web app
└── requirements.txt                    # Project dependencies
```

---

## 5. Installation & Quickstart

### 1. Clone & Setup Environment

```bash
git clone https://github.com/neitpm/Trafic.git
cd Trafic

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Reproduce Benchmarks & Training

#### Run Controlled Baseline:
```bash
python scripts/train/train_matched.py --model-type baseline --epochs 200 --batch 8 --imgsz 640 --seed 42
```

#### Run UIB-Nano (Scratch):
```bash
python scripts/train/train_matched.py --model-type uib_nano --epochs 200 --batch 8 --imgsz 640 --seed 42
```

#### Run UIB-Nano with Knowledge Distillation:
```bash
python scripts/train/train_distill.py --epochs 200 --batch 8 --imgsz 640 --seed 42 --dis 1.0
```

### 3. Run Hardware Latency Benchmark

Benchmark fused model throughput using hardware-synchronized `torch.cuda.Event` timing:

```bash
python scripts/benchmark_latency.py
```

### 4. Launch Interactive Streamlit Demo

Launch the traffic sign detection web app supporting Images, MP4 Videos, and Live Webcam streaming:

```bash
streamlit run app.py
```

---

## 6. Citation & License

This project is licensed under the [MIT License](LICENSE).  
Trained weights and architectures are intended for academic research, benchmarking, and prototyping.