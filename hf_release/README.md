---
license: mit
tags:
- object-detection
- yolo11
- traffic-sign-detection
- vietnamese-traffic-signs
- vrtsd
- computer-vision
- knowledge-distillation
- architecture-benchmark
datasets:
- custom
metrics:
- map
library_name: ultralytics
pipeline_tag: object-detection
---

# Vietnamese Traffic Sign Detection & Empirical Architecture Benchmark (VR-TSD-2)

This repository hosts reproducible model checkpoints, configuration manifests, and inference artifacts for the **Vietnamese Real-World Traffic Sign Detection Benchmark (VR-TSD-2)**.

The project evaluates architectural integration of **Universal Inverted Bottleneck (UIB)** modules into the **Ultralytics YOLO11** framework across 58 legal Vietnamese traffic sign categories (under national standard **QCVN 41:2019/BGTVT**).

---

## 1. Released Model Checkpoints

| Checkpoint Name | Model Architecture | Training Paradigm | Fused Params | FLOPs (640x640) | Mean Latency (RTX 5060 Ti) | mAP@[0.5:0.95] | Recommended Role |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `models/yolo11n_baseline_best.pt` | YOLO11n Baseline | COCO-Pretrained Fine-tune | **2.61M** | 6.5 G | **5.79 ms** | **70.27%** | **Recommended Benchmark & Demo Model** |
| `models/yolo11n_uib_nano_best.pt` | YOLO11n-UIB-Nano | Trained from Scratch | **2.48M** | **5.6 G** | 6.93 ms | 61.20% | Research Ablation (Compact Scratch) |
| `models/yolo11n_uib_nano_kd_best.pt`| YOLO11n-UIB-Nano + KD | Teacher-Student KD ($\text{dis}=1.0$) | **2.48M** | **5.6 G** | 6.93 ms | **64.05%** | Research Ablation (Best Distillation) |

*Historical variant note:* An unconstrained high-capacity variant (UIB 32M, 31.78M params, 51.9 GFLOPs) achieved 70.86% mAP@[0.5:0.95] from scratch, confirming representation capacity when parameters are unrestricted.

---

## 2. Benchmark Summary & Key Findings

1. **Recommended Model:** The **YOLO11n Baseline** is the recommended model for real-world deployment, achieving **70.27% mAP@[0.5:0.95]**, **92.19% Precision**, **81.44% Recall**, and **172.8 FPS** (5.79 ms) on an NVIDIA RTX 5060 Ti workstation.
2. **Compactness vs Pretraining:** **UIB-Nano** reduces parameters by 5.14% (2.48M) and FLOPs by 13.85% (5.6G). When trained from scratch without COCO pretraining, it achieves 61.20% mAP.
3. **Knowledge Distillation (KD):** Using the Baseline as a frozen teacher with optimal distillation weight ($\lambda_{\text{dis}} = 1.0$) improves UIB-Nano accuracy to **64.05% mAP** (+2.85 points), though a 6.22-point gap remains compared to the COCO-pretrained Baseline.
4. **Hardware Latency Reality:** While UIB-Nano reduces theoretical FLOPs, its GPU latency is ~1.20x higher (6.93 ms vs 5.79 ms) due to the Memory Access Cost (MAC) characteristic of Depthwise Separable Convolutions on workstation GPUs.
5. **Honest Scope:** UIB-Nano is **not claimed to be edge-optimized** or production-deployed; all latency figures reflect workstation GPU measurements.

---

## 3. Dataset & Legal Scope

- **Dataset:** Vietnamese Real-world Traffic Sign Dataset v2 (`VR-TSD-2`).
- **Classes:** 58 legal categories (Regulatory, Prohibitory, Warning, and Informative).
- **Split:** 5,668 train / 1,259 validation / 1,151 test.
- *Notice:* Raw dataset images and labels are **not redistributed** in this repository. All evaluation metrics are validation artifacts from the documented controlled protocol.

---

## 4. Quickstart: Inference with Ultralytics

### Installation
```bash
pip install ultralytics torch huggingface_hub
```

### Download Checkpoints
```bash
huggingface-cli download neitpm/traffic-sign-yolo11-benchmark --local-dir hf_release
```

### Python Inference
```python
from ultralytics import YOLO

# Load the recommended baseline model
model = YOLO("hf_release/models/yolo11n_baseline_best.pt")

# Predict on an image
results = model.predict("test_image.jpg", imgsz=640, conf=0.25, iou=0.45)
results[0].show()
```

---

## 5. Artifact Verification & Checksums

All files in this release are verified by SHA-256 signatures recorded in `checksums.sha256`.

---

## 6. License & Intended Use

This model repository is released under the [MIT License](LICENSE). Checkpoints and configurations are intended for research benchmarking, educational study, and deployment prototyping.
