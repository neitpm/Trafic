"""
Generate high-resolution 3D volumetric tensor block diagrams and architecture flows
for Vietnamese Traffic Sign Detection (VR-TSD) benchmark documentation.
Style: Palm-Vein-Recognition-Benchmark / OCR-Document style with isometric 3D blocks.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

# Ensure target directory
OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme as specified:
# blue = input
# orange = backbone
# green = UIB
# purple = neck/fusion
# red = detection/output
# gray = preprocessing/postprocessing
COLORS = {
    "blue": "#2563EB",
    "blue_top": "#60A5FA",
    "blue_side": "#1D4ED8",
    "orange": "#D97706",
    "orange_top": "#FBBF24",
    "orange_side": "#B45309",
    "green": "#059669",
    "green_top": "#34D399",
    "green_side": "#047857",
    "purple": "#7C3AED",
    "purple_top": "#A78BFA",
    "purple_side": "#6D28D9",
    "red": "#DC2626",
    "red_top": "#F87171",
    "red_side": "#B91C1C",
    "gray": "#4B5563",
    "gray_top": "#9CA3AF",
    "gray_side": "#374151",
    "dark_bg": "#0F172A",
    "card_bg": "#1E293B",
    "card_border": "#334155",
    "text_white": "#F8FAFC",
    "text_muted": "#94A3B8",
    "accent": "#38BDF8",
}

def draw_3d_box(ax, x, y, w, h, depth=0.4, color_theme="blue", label="", sublabel="", tensor_shape="", op_desc=""):
    """
    Draw a volumetric 3D isometric block with distinct front, top, and right facets.
    x, y: bottom-left coordinates of front face
    w, h: width and height of front face
    depth: isometric projection offset
    """
    front_c = COLORS[color_theme]
    top_c = COLORS[f"{color_theme}_top"]
    side_c = COLORS[f"{color_theme}_side"]
    
    dx = depth * 0.7
    dy = depth * 0.7
    
    # 1. Right Side facet
    side_pts = [
        [x + w, y],
        [x + w + dx, y + dy],
        [x + w + dx, y + h + dy],
        [x + w, y + h]
    ]
    ax.add_patch(Polygon(side_pts, closed=True, facecolor=side_c, edgecolor="#0F172A", lw=1.2, zorder=2))
    
    # 2. Top facet
    top_pts = [
        [x, y + h],
        [x + dx, y + h + dy],
        [x + w + dx, y + h + dy],
        [x + w, y + h]
    ]
    ax.add_patch(Polygon(top_pts, closed=True, facecolor=top_c, edgecolor="#0F172A", lw=1.2, zorder=2))
    
    # 3. Front facet
    front_pts = [
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ]
    ax.add_patch(Polygon(front_pts, closed=True, facecolor=front_c, edgecolor="#0F172A", lw=1.2, zorder=3))
    
    # Text annotations inside or on front face
    cx = x + w / 2.0
    cy = y + h / 2.0
    
    # Primary label (Component Name)
    if label:
        ax.text(cx, cy + h * 0.22, label, ha="center", va="center", color="#FFFFFF",
                fontsize=9.5, fontweight="bold", zorder=4)
    # Sublabel (Layer / Type)
    if sublabel:
        ax.text(cx, cy + h * 0.02, sublabel, ha="center", va="center", color="#F1F5F9",
                fontsize=8.0, fontweight="medium", zorder=4)
    # Tensor Shape
    if tensor_shape:
        ax.text(cx, cy - h * 0.22, tensor_shape, ha="center", va="center", color="#FEF08A",
                fontsize=7.8, fontweight="bold", family="monospace", zorder=4)
        
    # Operation description placed above or below block
    if op_desc:
        ax.text(cx + dx/2, y - 0.28, op_desc, ha="center", va="top", color="#94A3B8",
                fontsize=7.5, style="italic", zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, color="#38BDF8", text="", text_offset_y=0.18):
    """Draw a clean directional connection arrow with optional text."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(facecolor=color, edgecolor=color, width=1.8, headwidth=6, headlength=7, shrink=0.08),
        zorder=5
    )
    if text:
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0 + text_offset_y
        ax.text(mid_x, mid_y, text, ha="center", va="center", color="#E2E8F0",
                fontsize=7.5, fontweight="bold", backgroundcolor="#0F172A", zorder=6)


# ==============================================================================
# Figure 1: 3D End-to-End Traffic Sign Detection Pipeline
# ==============================================================================
def generate_figure_1():
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    fig.patch.set_facecolor(COLORS["dark_bg"])
    ax.set_facecolor(COLORS["dark_bg"])
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-1.0, 5.0)
    ax.axis("off")
    
    ax.text(7.5, 4.4, "End-to-End Traffic Sign Detection Pipeline", ha="center", va="center",
            color=COLORS["text_white"], fontsize=17, fontweight="bold")
    ax.text(7.5, 3.95, "VR-TSD Dataset (58 Classes) | Raw Image to Calibrated Bounding Box & Class ID",
            ha="center", va="center", color=COLORS["text_muted"], fontsize=10.5)
    
    stages = [
        {"x": 0.2, "w": 1.7, "h": 2.2, "theme": "blue", "label": "Input Image", "sub": "Raw Frame", "shape": "(3, H, W)", "desc": "RGB Camera Feed"},
        {"x": 2.6, "w": 1.7, "h": 2.2, "theme": "gray", "label": "Letterbox", "sub": "Prep & Norm", "shape": "(3, 640, 640)", "desc": "Pad / Scale / [0,1]"},
        {"x": 5.0, "w": 2.1, "h": 2.2, "theme": "orange", "label": "Backbone", "sub": "Feature Extractor", "shape": "P3, P4, P5", "desc": "Hierarchical Repr."},
        {"x": 7.8, "w": 2.1, "h": 2.2, "theme": "purple", "label": "PAN / Neck", "sub": "Bi-Path Fusion", "shape": "Multiscale Pyramids", "desc": "Top-Down & Bottom-Up"},
        {"x": 10.6, "w": 1.9, "h": 2.2, "theme": "red", "label": "Detect Head", "sub": "Decoupled Head", "shape": "(B, 64+58, 8400)", "desc": "Cls & Reg Projections"},
        {"x": 13.2, "w": 1.8, "h": 2.2, "theme": "gray", "label": "NMS / Output", "sub": "Postprocessing", "shape": "(N, [x,y,w,h,conf,cls])", "desc": "IoU=0.45, Conf=0.25"}
    ]
    
    for s in stages:
        draw_3d_box(ax, s["x"], 0.7, s["w"], s["h"], depth=0.45, color_theme=s["theme"],
                    label=s["label"], sublabel=s["sub"], tensor_shape=s["shape"], op_desc=s["desc"])
        
    for i in range(len(stages) - 1):
        x1 = stages[i]["x"] + stages[i]["w"] + 0.35
        y1 = 0.7 + stages[i]["h"] / 2.0 + 0.15
        x2 = stages[i+1]["x"]
        y2 = y1
        draw_arrow(ax, x1, y1, x2, y2, color=COLORS["accent"])
        
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "1_pipeline.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("Saved 1_pipeline.png")


# ==============================================================================
# Figure 2: YOLO11n Baseline 3D Architecture
# ==============================================================================
def generate_figure_2():
    fig, ax = plt.subplots(figsize=(16, 8.0), dpi=300)
    fig.patch.set_facecolor(COLORS["dark_bg"])
    ax.set_facecolor(COLORS["dark_bg"])
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-1.5, 6.2)
    ax.axis("off")
    
    ax.text(8.0, 5.6, "YOLO11n Baseline Architecture (COCO-Pretrained)", ha="center", va="center",
            color=COLORS["text_white"], fontsize=17, fontweight="bold")
    ax.text(8.0, 5.1, "Parameters: 2.61M | FLOPs: 6.5G | mAP@[0.5:0.95]: 70.27% | Latency: 5.79 ms (RTX 5060 Ti)",
            ha="center", va="center", color=COLORS["text_muted"], fontsize=10.5)
    
    # Row 1: Backbone stages (Bottom-up)
    bb_stages = [
        {"x": 0.2, "w": 1.7, "h": 1.8, "theme": "orange", "label": "Conv Stem", "sub": "k=3, s=2", "shape": "(16, 320, 320)", "desc": "Stride 2"},
        {"x": 2.4, "w": 1.7, "h": 1.8, "theme": "orange", "label": "Stage 1", "sub": "Conv + C3k2", "shape": "(32, 160, 160)", "desc": "P2 Stride 4"},
        {"x": 4.6, "w": 1.7, "h": 1.8, "theme": "orange", "label": "Stage 2 (P3)", "sub": "Conv + C3k2", "shape": "(64, 80, 80)", "desc": "P3 Stride 8"},
        {"x": 6.8, "w": 1.7, "h": 1.8, "theme": "orange", "label": "Stage 3 (P4)", "sub": "Conv + C3k2", "shape": "(128, 40, 40)", "desc": "P4 Stride 16"},
        {"x": 9.0, "w": 1.7, "h": 1.8, "theme": "orange", "label": "Stage 4 (P5)", "sub": "Conv + C3k2", "shape": "(256, 20, 20)", "desc": "P5 Stride 32"},
        {"x": 11.2, "w": 1.7, "h": 1.8, "theme": "orange", "label": "SPPF", "sub": "Spatial Pyramid", "shape": "(256, 20, 20)", "desc": "k=5 Pooling"},
        {"x": 13.4, "w": 1.7, "h": 1.8, "theme": "orange", "label": "C2PSA", "sub": "Self-Attention", "shape": "(256, 20, 20)", "desc": "Pointwise SA"}
    ]
    
    for s in bb_stages:
        draw_3d_box(ax, s["x"], 3.0, s["w"], s["h"], depth=0.35, color_theme=s["theme"],
                    label=s["label"], sublabel=s["sub"], tensor_shape=s["shape"], op_desc="")
        
    for i in range(len(bb_stages) - 1):
        x1 = bb_stages[i]["x"] + bb_stages[i]["w"] + 0.28
        y1 = 3.0 + bb_stages[i]["h"] / 2.0 + 0.1
        x2 = bb_stages[i+1]["x"]
        y2 = y1
        draw_arrow(ax, x1, y1, x2, y2, color=COLORS["orange_top"])
        
    neck_stages = [
        {"x": 3.0, "w": 2.4, "h": 1.8, "theme": "purple", "label": "P3 Neck Fusion", "sub": "Concat + C3k2", "shape": "(64, 80, 80)", "desc": "Small Signs"},
        {"x": 6.8, "w": 2.4, "h": 1.8, "theme": "purple", "label": "P4 Neck Fusion", "sub": "Concat + C3k2", "shape": "(128, 40, 40)", "desc": "Medium Signs"},
        {"x": 10.6, "w": 2.4, "h": 1.8, "theme": "purple", "label": "P5 Neck Fusion", "sub": "Concat + C3k2", "shape": "(256, 20, 20)", "desc": "Large Signs"}
    ]
    
    for s in neck_stages:
        draw_3d_box(ax, s["x"], -0.4, s["w"], s["h"], depth=0.35, color_theme=s["theme"],
                    label=s["label"], sublabel=s["sub"], tensor_shape=s["shape"], op_desc=s["desc"])
        
    # Connect Backbone to Neck with high clearance
    draw_arrow(ax, 5.45, 2.9, 4.2, 1.6, color=COLORS["purple_top"], text="P3 Feat", text_offset_y=0.15)
    draw_arrow(ax, 7.65, 2.9, 8.0, 1.6, color=COLORS["purple_top"], text="P4 Feat", text_offset_y=0.15)
    draw_arrow(ax, 14.25, 2.9, 11.8, 1.6, color=COLORS["purple_top"], text="P5 Feat", text_offset_y=0.15)
    
    # Detect Head box
    draw_3d_box(ax, 13.6, -0.4, 2.2, 1.8, depth=0.35, color_theme="red",
                label="Detect Head", sublabel="Decoupled 58 cls", tensor_shape="P3, P4, P5 heads", op_desc="mAP: 70.27%")
    draw_arrow(ax, 13.2, 0.6, 13.6, 0.6, color=COLORS["red_top"])
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "2_yolo11_baseline_3d.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("Saved 2_yolo11_baseline_3d.png")


# ==============================================================================
# Figure 3: UIB Block Decomposition
# ==============================================================================
def generate_figure_3():
    fig, ax = plt.subplots(figsize=(16, 7.0), dpi=300)
    fig.patch.set_facecolor(COLORS["dark_bg"])
    ax.set_facecolor(COLORS["dark_bg"])
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-1.5, 6.0)
    ax.axis("off")
    
    ax.text(8.0, 5.3, "Universal Inverted Bottleneck (UIB) Block Decomposition", ha="center", va="center",
            color=COLORS["text_white"], fontsize=17, fontweight="bold")
    ax.text(8.0, 4.8, "Expansion 1x1 → Depthwise Separable 3x3 → Squeeze-and-Excitation (SE) → Linear 1x1 Projection + Residual",
            ha="center", va="center", color=COLORS["text_muted"], fontsize=10.5)
    
    uib_steps = [
        {"x": 0.2, "w": 1.8, "h": 2.2, "theme": "blue", "label": "Input Tensor", "sub": "X", "shape": "(C1, H, W)", "desc": "Incoming features"},
        {"x": 2.6, "w": 2.1, "h": 2.2, "theme": "green", "label": "ConvBNAct (1x1)", "sub": "Expansion e=2 or 6", "shape": "(hidden, H, W)", "desc": "Pointwise Expand"},
        {"x": 5.3, "w": 2.2, "h": 2.2, "theme": "green", "label": "Depthwise Conv", "sub": "k=3, s=1 or 2", "shape": "(hidden, H', W')", "desc": "Spatial Context"},
        {"x": 8.1, "w": 2.2, "h": 2.2, "theme": "green", "label": "Squeeze-Excite", "sub": "SE (r=0.25)", "shape": "(hidden, H', W')", "desc": "Channel Attention"},
        {"x": 10.9, "w": 2.1, "h": 2.2, "theme": "green", "label": "ConvBNAct (1x1)", "sub": "Projection (No Act)", "shape": "(C2, H', W')", "desc": "Linear Projection"},
        {"x": 13.6, "w": 1.9, "h": 2.2, "theme": "red", "label": "Output Tensor", "sub": "Y = F(X) + X", "shape": "(C2, H', W')", "desc": "Residual fused if s=1"}
    ]
    
    for s in uib_steps:
        draw_3d_box(ax, s["x"], 0.7, s["w"], s["h"], depth=0.4, color_theme=s["theme"],
                    label=s["label"], sublabel=s["sub"], tensor_shape=s["shape"], op_desc=s["desc"])
        
    for i in range(len(uib_steps) - 1):
        x1 = uib_steps[i]["x"] + uib_steps[i]["w"] + 0.35
        y1 = 0.7 + uib_steps[i]["h"] / 2.0 + 0.15
        x2 = uib_steps[i+1]["x"]
        y2 = y1
        draw_arrow(ax, x1, y1, x2, y2, color=COLORS["accent"])
        
    # Residual Skip Connection (Arc cleanly placed above)
    ax.annotate(
        "", xy=(14.2, 3.3), xytext=(1.2, 3.3),
        arrowprops=dict(arrowstyle="->", color="#34D399", lw=2.4, connectionstyle="arc3,rad=-0.12"),
        zorder=5
    )
    ax.text(7.7, 3.95, "Identity Residual Connection (if stride=1 and C1 == C2)", ha="center", va="center",
            color="#34D399", fontsize=9.5, fontweight="bold", backgroundcolor=COLORS["dark_bg"], zorder=6)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "3_uib_block_3d.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("Saved 3_uib_block_3d.png")


# ==============================================================================
# Figure 4: UIB-Nano 3D Architecture
# ==============================================================================
def generate_figure_4():
    fig, ax = plt.subplots(figsize=(16, 8.0), dpi=300)
    fig.patch.set_facecolor(COLORS["dark_bg"])
    ax.set_facecolor(COLORS["dark_bg"])
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-1.5, 6.2)
    ax.axis("off")
    
    ax.text(8.0, 5.6, "YOLO11n-UIB-Nano 3D Architecture (2.48M Lightweight Backbone)", ha="center", va="center",
            color=COLORS["text_white"], fontsize=17, fontweight="bold")
    ax.text(8.0, 5.1, "Parameters: 2.48M (-5.1%) | FLOPs: 5.6G (-13.8%) | UIB expansion e=2.0 | P3, P4, P5 Detect Heads",
            ha="center", va="center", color=COLORS["text_muted"], fontsize=10.5)
    
    bb_stages = [
        {"x": 0.2, "w": 1.7, "h": 1.8, "theme": "green", "label": "Conv Stem", "sub": "k=3, s=2", "shape": "(16, 320, 320)", "desc": "Layer 0"},
        {"x": 2.4, "w": 1.7, "h": 1.8, "theme": "green", "label": "UIB P2", "sub": "UIBDown+UIB", "shape": "(32, 160, 160)", "desc": "Layers 1-2"},
        {"x": 4.6, "w": 1.7, "h": 1.8, "theme": "green", "label": "UIB P3", "sub": "UIBDown+UIB", "shape": "(64, 80, 80)", "desc": "Layers 3-4 (P3)"},
        {"x": 6.8, "w": 1.7, "h": 1.8, "theme": "green", "label": "UIB P4", "sub": "UIBDown+UIB", "shape": "(128, 40, 40)", "desc": "Layers 5-6 (P4)"},
        {"x": 9.0, "w": 1.7, "h": 1.8, "theme": "green", "label": "UIB P5", "sub": "UIBDown+UIB", "shape": "(256, 20, 20)", "desc": "Layers 7-8 (P5)"},
        {"x": 11.2, "w": 1.7, "h": 1.8, "theme": "orange", "label": "SPPF", "sub": "k=5, 256ch", "shape": "(256, 20, 20)", "desc": "Layer 9"},
        {"x": 13.4, "w": 1.7, "h": 1.8, "theme": "orange", "label": "C2PSA", "sub": "Self-Attn 256ch", "shape": "(256, 20, 20)", "desc": "Layer 10"}
    ]
    
    for s in bb_stages:
        draw_3d_box(ax, s["x"], 3.0, s["w"], s["h"], depth=0.35, color_theme=s["theme"],
                    label=s["label"], sublabel=s["sub"], tensor_shape=s["shape"], op_desc="")
        
    for i in range(len(bb_stages) - 1):
        x1 = bb_stages[i]["x"] + bb_stages[i]["w"] + 0.28
        y1 = 3.0 + bb_stages[i]["h"] / 2.0 + 0.1
        x2 = bb_stages[i+1]["x"]
        y2 = y1
        draw_arrow(ax, x1, y1, x2, y2, color=COLORS["green_top"])
        
    neck_stages = [
        {"x": 3.0, "w": 2.4, "h": 1.8, "theme": "purple", "label": "P3 Neck Fusion", "sub": "Concat + C3k2", "shape": "(64, 80, 80)", "desc": "Layer 16"},
        {"x": 6.8, "w": 2.4, "h": 1.8, "theme": "purple", "label": "P4 Neck Fusion", "sub": "Concat + C3k2", "shape": "(128, 40, 40)", "desc": "Layer 19"},
        {"x": 10.6, "w": 2.4, "h": 1.8, "theme": "purple", "label": "P5 Neck Fusion", "sub": "Concat + C3k2", "shape": "(256, 20, 20)", "desc": "Layer 22"}
    ]
    
    for s in neck_stages:
        draw_3d_box(ax, s["x"], -0.4, s["w"], s["h"], depth=0.35, color_theme=s["theme"],
                    label=s["label"], sublabel=s["sub"], tensor_shape=s["shape"], op_desc=s["desc"])
        
    draw_arrow(ax, 5.45, 2.9, 4.2, 1.6, color=COLORS["purple_top"], text="P3 Feat (L4)", text_offset_y=0.15)
    draw_arrow(ax, 7.65, 2.9, 8.0, 1.6, color=COLORS["purple_top"], text="P4 Feat (L6)", text_offset_y=0.15)
    draw_arrow(ax, 14.25, 2.9, 11.8, 1.6, color=COLORS["purple_top"], text="P5 Feat (L10)", text_offset_y=0.15)
    
    draw_3d_box(ax, 13.6, -0.4, 2.2, 1.8, depth=0.35, color_theme="red",
                label="3-Scale Detect", sublabel="P3, P4, P5 (nc=58)", tensor_shape="Layer 23", op_desc="mAP: 61.2% (Scratch)")
    draw_arrow(ax, 13.2, 0.6, 13.6, 0.6, color=COLORS["red_top"])
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "4_uib_nano_3d.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("Saved 4_uib_nano_3d.png")


# ==============================================================================
# Figure 5: Knowledge Distillation Pipeline
# ==============================================================================
def generate_figure_5():
    fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300)
    fig.patch.set_facecolor(COLORS["dark_bg"])
    ax.set_facecolor(COLORS["dark_bg"])
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-1.5, 6.0)
    ax.axis("off")
    
    ax.text(8.0, 5.4, "Teacher-Student Knowledge Distillation Pipeline", ha="center", va="center",
            color=COLORS["text_white"], fontsize=17, fontweight="bold")
    ax.text(8.0, 4.9, "Teacher (Frozen YOLO11n Baseline) → Distillation Loss (dis=1.0) → Student (UIB-Nano) + Detection Loss",
            ha="center", va="center", color=COLORS["text_muted"], fontsize=10.5)
    
    draw_3d_box(ax, 0.2, 1.5, 1.8, 2.2, depth=0.35, color_theme="blue",
                label="Input Batch", sublabel="Training Pair", tensor_shape="(8, 3, 640, 640)", op_desc="VR-TSD 58 classes")
    
    draw_3d_box(ax, 3.2, 2.8, 3.2, 1.8, depth=0.35, color_theme="orange",
                label="Teacher: YOLO11n Baseline", sublabel="Frozen Pretrained", tensor_shape="P3, P4, P5 Logits/Feats", op_desc="mAP: 70.27%")
    
    draw_3d_box(ax, 3.2, 0.2, 3.2, 1.8, depth=0.35, color_theme="green",
                label="Student: UIB-Nano", sublabel="Trainable 2.48M", tensor_shape="P3, P4, P5 Logits/Feats", op_desc="Scratch: 61.20%")
    
    draw_arrow(ax, 2.2, 2.8, 3.2, 3.7, color=COLORS["accent"], text="No Grad Forward")
    draw_arrow(ax, 2.2, 2.4, 3.2, 1.1, color=COLORS["accent"], text="Grad Forward")
    
    draw_3d_box(ax, 7.8, 2.8, 3.0, 1.8, depth=0.35, color_theme="purple",
                label="Distillation Loss", sublabel="L_KD (MSE / CW3)", tensor_shape="Feature Alignment", op_desc="Weight dis=1.0")
    
    draw_3d_box(ax, 7.8, 0.2, 3.0, 1.8, depth=0.35, color_theme="red",
                label="Task Detection Loss", sublabel="L_task (Box + Cls + DFL)", tensor_shape="vs Ground Truth", op_desc="Standard YOLO Loss")
    
    draw_arrow(ax, 6.7, 3.7, 7.8, 3.7, color=COLORS["orange_top"], text="Teacher Output")
    draw_arrow(ax, 6.7, 1.8, 7.8, 3.2, color=COLORS["green_top"], text="Student Logits")
    draw_arrow(ax, 6.7, 1.1, 7.8, 1.1, color=COLORS["green_top"], text="Student Preds")
    
    draw_3d_box(ax, 12.4, 1.5, 3.2, 2.2, depth=0.35, color_theme="gray",
                label="Total Loss Optimization", sublabel="L = L_task + dis * L_KD", tensor_shape="AdamW Optimizer", op_desc="Best: 64.05% mAP50-95")
    
    draw_arrow(ax, 11.0, 3.7, 12.4, 2.9, color=COLORS["purple_top"], text="L_KD")
    draw_arrow(ax, 11.0, 1.1, 12.4, 2.3, color=COLORS["red_top"], text="L_task")
    
    # Backpropagation path routed beneath blocks cleanly
    # (14.0, 1.3) -> down to (14.0, -0.6) -> left to (4.8, -0.6) -> up to (4.8, 0.1)
    ax.annotate(
        "", xy=(4.8, 0.1), xytext=(14.0, 1.3),
        arrowprops=dict(
            arrowstyle="->", color="#F87171", lw=2.2,
            connectionstyle="angle,angleA=-90,angleB=180,rad=15"
        ),
        zorder=5
    )
    ax.text(9.4, -0.85, "Backpropagation to Student Parameters Only", ha="center", va="center",
            color="#F87171", fontsize=8.5, fontweight="bold", backgroundcolor=COLORS["dark_bg"], zorder=6)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "5_distillation_pipeline_3d.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("Saved 5_distillation_pipeline_3d.png")


# ==============================================================================
# Figure 6: Streamlit Inference Pipeline
# ==============================================================================
def generate_figure_6():
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    fig.patch.set_facecolor(COLORS["dark_bg"])
    ax.set_facecolor(COLORS["dark_bg"])
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-1.0, 5.0)
    ax.axis("off")
    
    ax.text(7.5, 4.4, "Streamlit Real-Time Inference Architecture", ha="center", va="center",
            color=COLORS["text_white"], fontsize=17, fontweight="bold")
    ax.text(7.5, 3.95, "Dockerized Containerized Web App | Dynamic Model Preset Selector | Image/Video/Webcam Modes",
            ha="center", va="center", color=COLORS["text_muted"], fontsize=10.5)
    
    stages = [
        {"x": 0.2, "w": 1.8, "h": 2.2, "theme": "blue", "label": "User Input", "sub": "Upload / Webcam", "shape": "JPG/PNG/MP4", "desc": "Browser UI"},
        {"x": 2.7, "w": 1.9, "h": 2.2, "theme": "gray", "label": "Model Preset", "sub": "Weights Selector", "shape": "TRAFFIC_SIGN_WEIGHTS", "desc": "Baseline / UIB-Nano"},
        {"x": 5.2, "w": 2.1, "h": 2.2, "theme": "orange", "label": "Inference Core", "sub": "YOLO / UIB Engine", "shape": "Fused Latency: 5.79ms", "desc": "CPU / CUDA Backend"},
        {"x": 7.9, "w": 2.0, "h": 2.2, "theme": "purple", "label": "Thresholding", "sub": "Conf & IoU Filter", "shape": "conf=0.25, iou=0.45", "desc": "Interactive Sliders"},
        {"x": 10.5, "w": 2.1, "h": 2.2, "theme": "green", "label": "Class Mapping", "sub": "VR-TSD Dictionary", "shape": "58 Vietnamese Classes", "desc": "Legal Category IDs"},
        {"x": 13.1, "w": 1.9, "h": 2.2, "theme": "red", "label": "Render Display", "sub": "Interactive Canvas", "shape": "Bounding Boxes + FPS", "desc": "Overlay & Metadata"}
    ]
    
    for s in stages:
        draw_3d_box(ax, s["x"], 0.7, s["w"], s["h"], depth=0.4, color_theme=s["theme"],
                    label=s["label"], sublabel=s["sub"], tensor_shape=s["shape"], op_desc=s["desc"])
        
    for i in range(len(stages) - 1):
        x1 = stages[i]["x"] + stages[i]["w"] + 0.35
        y1 = 0.7 + stages[i]["h"] / 2.0 + 0.15
        x2 = stages[i+1]["x"]
        y2 = y1
        draw_arrow(ax, x1, y1, x2, y2, color=COLORS["accent"])
        
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "6_streamlit_pipeline_3d.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("Saved 6_streamlit_pipeline_3d.png")


# ==============================================================================
# Figure 7: Model Comparison (Pareto Analysis & Metric Trade-offs)
# ==============================================================================
def generate_figure_7():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)
    fig.patch.set_facecolor(COLORS["dark_bg"])
    
    for ax in (ax1, ax2):
        ax.set_facecolor(COLORS["card_bg"])
        ax.tick_params(colors=COLORS["text_muted"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(COLORS["card_border"])
            
    fig.suptitle("Empirical Model Benchmark & Pareto Analysis (VR-TSD-2)",
                 color=COLORS["text_white"], fontsize=17, fontweight="bold", y=0.98)
    
    models = [
        {"name": "YOLO11n Baseline\n(Recommended)", "params": 2.61, "map": 70.27, "lat": 5.79, "c": "#38BDF8", "marker": "*", "s": 320},
        {"name": "UIB-Nano Scratch", "params": 2.48, "map": 61.20, "lat": 6.93, "c": "#F87171", "marker": "o", "s": 150},
        {"name": "UIB-Nano + KD (dis=0.5)", "params": 2.48, "map": 63.59, "lat": 6.93, "c": "#FBBF24", "marker": "s", "s": 160},
        {"name": "UIB-Nano + KD (dis=1.0)", "params": 2.48, "map": 64.05, "lat": 6.93, "c": "#34D399", "marker": "^", "s": 200},
        {"name": "UIB-Nano + KD (dis=2.0)", "params": 2.48, "map": 62.08, "lat": 6.93, "c": "#A78BFA", "marker": "v", "s": 150},
        {"name": "UIB Original (32M)", "params": 31.78, "map": 70.86, "lat": 15.20, "c": "#9CA3AF", "marker": "D", "s": 180}
    ]
    
    # Plot Subplot 1 (Params vs mAP)
    for m in models:
        ax1.scatter(m["params"], m["map"], color=m["c"], s=m["s"], marker=m["marker"], edgecolor="#0F172A", lw=1.5, zorder=4)
        
    ax1.annotate("YOLO11n Baseline\n(Recommended: 70.27% mAP, 2.61M)", xy=(2.61, 70.27), xytext=(4.0, 68.5),
                 color="#38BDF8", fontsize=9.0, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=1.5))
    
    ax1.annotate("UIB-Nano + KD (dis=1.0)\n(Best KD: 64.05% mAP, 2.48M)", xy=(2.48, 64.05), xytext=(4.0, 63.5),
                 color="#34D399", fontsize=9.0, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#34D399", lw=1.5))
    
    ax1.annotate("UIB-Nano Scratch\n(61.20% mAP)", xy=(2.48, 61.20), xytext=(4.0, 60.5),
                 color="#F87171", fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="#F87171", lw=1.2))
    
    ax1.annotate("UIB 32M (Historical)\n(70.86% mAP, 31.8M)", xy=(31.78, 70.86), xytext=(22.0, 68.0),
                 color="#9CA3AF", fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="#9CA3AF", lw=1.2))
    
    ax1.set_xlabel("Parameters (Million)", color=COLORS["text_white"], fontsize=11, fontweight="bold")
    ax1.set_ylabel("Validation mAP@[0.5:0.95] (%)", color=COLORS["text_white"], fontsize=11, fontweight="bold")
    ax1.set_title("Capacity vs Accuracy Trade-off", color=COLORS["text_white"], fontsize=12, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.25, color=COLORS["card_border"])
    ax1.set_ylim(58, 73)
    ax1.set_xlim(1.5, 34)
    
    # Subplot 2: Latency vs mAP
    for m in models:
        ax2.scatter(m["lat"], m["map"], color=m["c"], s=m["s"], marker=m["marker"], edgecolor="#0F172A", lw=1.5, zorder=4)
        
    ax2.annotate("Baseline: Lower Latency\n(5.79 ms / 172.8 FPS)", xy=(5.79, 70.27), xytext=(6.5, 71.2),
                 color="#38BDF8", fontsize=9.0, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=1.5))
    
    ax2.annotate("UIB-Nano: Depthwise MAC Penalty\n(6.93 ms / 144.4 FPS)", xy=(6.93, 64.05), xytext=(8.0, 64.0),
                 color="#34D399", fontsize=9.0, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#34D399", lw=1.5))
    
    ax2.set_xlabel("Fused Latency on RTX 5060 Ti (ms)", color=COLORS["text_white"], fontsize=11, fontweight="bold")
    ax2.set_ylabel("Validation mAP@[0.5:0.95] (%)", color=COLORS["text_white"], fontsize=11, fontweight="bold")
    ax2.set_title("Latency vs Accuracy (Workstation GPU)", color=COLORS["text_white"], fontsize=12, fontweight="bold", pad=12)
    ax2.grid(True, linestyle="--", alpha=0.25, color=COLORS["card_border"])
    ax2.set_ylim(58, 73)
    ax2.set_xlim(4.5, 17.5)
    
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='#38BDF8', label='YOLO11n Baseline (Recommended Model)', markersize=10, linestyle='None'),
        plt.Line2D([0], [0], marker='^', color='#34D399', label='UIB-Nano + KD (dis=1.0 - Best Distillation)', markersize=8, linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='#F87171', label='UIB-Nano Scratch (Ablation Baseline)', markersize=8, linestyle='None'),
        plt.Line2D([0], [0], marker='D', color='#9CA3AF', label='UIB 32M (High Capacity Research)', markersize=8, linestyle='None')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, frameon=True,
               facecolor=COLORS["card_bg"], edgecolor=COLORS["card_border"], labelcolor=COLORS["text_white"],
               fontsize=9, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    fig.savefig(OUTPUT_DIR / "7_model_comparison.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("Saved 7_model_comparison.png")

def main():
    print(f"Generating 3D architectural diagrams into {OUTPUT_DIR.resolve()}...")
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    generate_figure_5()
    generate_figure_6()
    generate_figure_7()
    print("All 7 architectural and benchmark diagrams successfully generated at 300 DPI!")

if __name__ == "__main__":
    main()
