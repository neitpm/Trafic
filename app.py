import os
import gc
import sys
import time
import random
import argparse
import tempfile
import subprocess
from pathlib import Path

import cv2
import torch
import numpy as np
import streamlit as st
from ultralytics import YOLO



PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = Path(os.getenv("TRAFFIC_SIGN_WEIGHTS", str(PROJECT_ROOT / "weights" / "yolo11n_uib_best.pt")))
ALT_WEIGHTS = PROJECT_ROOT / "pretrain_backbone_yolo11n" / "train_with_colab" / "vrtsd_uib_p2_fix2" / "weights" / "best.pt"
PRED_DIR = PROJECT_ROOT / "outputs" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

VRTSD_CLASS_NAMES = {
    0: "101-duong-cam",
    1: "102-cam-di-nguoc-chieu",
    2: "103a-cam-oto",
    3: "103b-cam-oto-re-phai",
    4: "103c-cam-oto-re-trai",
    5: "106-cam-oto-tai",
    6: "107-cam-oto-khach-va-oto-tai",
    7: "123a-cam-re-trai",
    8: "123b-cam-re-phai",
    9: "124a-cam-quay-dau-xe",
    10: "124b-cam-oto-quay-dau-xe",
    11: "125-cam-vuot",
    12: "127-toc-do-toi-da-40",
    13: "127-toc-do-toi-da-50",
    14: "127-toc-do-toi-da-60",
    15: "127-toc-do-toi-da-80",
    16: "127-toc-do-toi-da-100",
    17: "127-toc-do-toi-da-120",
    18: "129-kiem-tra",
    19: "130-cam-dung-xe-va-do-xe",
    20: "131a-cam-do-xe",
    21: "131b-cam-do-xe-ngay-le",
    22: "131c-cam-do-xe-ngay-chan",
    23: "132-nhuong-duong",
    24: "134-cam-re-trai",
    25: "135-cam-re-phai",
    26: "136-cam-di-thang",
    27: "137-cam-re-trai-va-re-phai",
    28: "138-cam-di-thang-va-re-trai",
    29: "139-cam-di-thang-va-re-phai",
    30: "140-cam-xe-cong-nong",
    31: "203-duong-bi-hep-ca-hai-ben",
    32: "204-duong-hai-chieu",
    33: "205a-duong-giao-nhau-cung-muc",
    34: "205b-duong-giao-nhau-cung-muc",
    35: "205c-duong-giao-nhau-cung-muc",
    36: "208-giao-nhau-voi-duong-uu-tien",
    37: "209-giao-nhau-co-tin-hieu-den",
    38: "225-duong-nguoi-di-bo-cat-ngang",
    39: "227-cong-truong",
    40: "228-da-vang",
    41: "229-duong-may-bay-cat-ngang",
    42: "230-nguoi-mu-cat-ngang",
    43: "231-tre-em",
    44: "233-nguy-hiem-khac",
    45: "234-giao-nhau-voi-duong-sat-khong-rao-chan",
    46: "235-duong-do-doc-nguy-hiem",
    47: "237-cau-hep",
    48: "243a-noi-duong-sat-cat-duong-bo",
    49: "245-di-cham",
    50: "302a-huong-phai-di-vong-chuong-ngai-vat",
    51: "303-cam-xe-co-gioi-di-nguoc-chieu",
    52: "407a-duong-mot-chieu",
    53: "409-cho-quay-xe",
    54: "421-ket-thuc-khu-dong-dan-cu",
    55: "423a-duong-nguoi-di-bo-sang-ngang-1",
    56: "437-bat-dau-duong-cao-toc",
    57: "438-ket-thuc-duong-cao-toc",
}

class ConvBNAct(torch.nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        p = k // 2
        self.conv = torch.nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = torch.nn.BatchNorm2d(c2)
        self.act = torch.nn.SiLU(inplace=True) if act else torch.nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SqueezeExcite(torch.nn.Module):
    def __init__(self, c, r=0.25):
        super().__init__()
        hidden = max(8, int(c * r))
        self.avg = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(c, hidden, 1, bias=True),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(hidden, c, 1, bias=True),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w


class UIB(torch.nn.Module):
    def __init__(self, c1, c2, k=3, s=1, e=6.0):
        super().__init__()
        hidden = max(16, int(round(c1 * e)))
        self.use_res = (s == 1 and c1 == c2)
        self.block = torch.nn.Sequential(
            ConvBNAct(c1, hidden, k=1, s=1, act=True),
            ConvBNAct(hidden, hidden, k=k, s=s, g=hidden, act=True),
            SqueezeExcite(hidden, r=0.25),
            ConvBNAct(hidden, c2, k=1, s=1, act=False),
        )

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class UIBDown(torch.nn.Module):
    def __init__(self, c1, c2, k=3, s=2, e=6.0):
        super().__init__()
        hidden = max(16, int(round(c1 * e)))
        self.block = torch.nn.Sequential(
            ConvBNAct(c1, hidden, k=1, s=1, act=True),
            ConvBNAct(hidden, hidden, k=k, s=s, g=hidden, act=True),
            SqueezeExcite(hidden, r=0.25),
            ConvBNAct(hidden, c2, k=1, s=1, act=False),
        )

    def forward(self, x):
        return self.block(x)


def register_uib_modules():
    import ultralytics.nn.tasks as tasks

    tasks.UIB = UIB
    tasks.UIBDown = UIBDown

    try:
        tasks.parse_model.__globals__["UIB"] = UIB
        tasks.parse_model.__globals__["UIBDown"] = UIBDown
    except Exception:
        pass

    try:
        tasks.__dict__["UIB"] = UIB
        tasks.__dict__["UIBDown"] = UIBDown
    except Exception:
        pass


def get_default_weights():
    if DEFAULT_WEIGHTS.exists():
        return str(DEFAULT_WEIGHTS)
    if ALT_WEIGHTS.exists():
        return str(ALT_WEIGHTS)
    return ""


def get_color(class_id: int):
    palette = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
        (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
        (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
    ]
    return palette[class_id % len(palette)]


def draw_label(img, text, x1, y1, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    y_top = max(0, y1 - th - baseline - 8)
    y_text = max(th + 2, y1 - 5)

    cv2.rectangle(img, (x1, y_top), (x1 + tw + 8, y1), color, -1)
    cv2.putText(
        img,
        text,
        (x1 + 4, y_text),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_fps_panel(img, fps, num_det):
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (250, 75), (25, 25, 25), -1)
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    cv2.putText(img, f"FPS: {fps:.1f}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f"Detections: {num_det}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def normalize_names_obj(names):
    if names is None:
        return {}

    if isinstance(names, dict):
        out = {}
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                pass
        return out

    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}

    return {}


def get_class_name(result, model, cls_id: int):
    cls_id = int(cls_id)

    result_names = normalize_names_obj(getattr(result, "names", None))
    if cls_id in result_names:
        name = result_names[cls_id]
        if name and not name.startswith("class_"):
            return name

    model_names = normalize_names_obj(getattr(model, "names", None))
    if cls_id in model_names:
        name = model_names[cls_id]
        if name and not name.startswith("class_"):
            return name

    return VRTSD_CLASS_NAMES.get(cls_id, f"class_{cls_id}")


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    import __main__

    __main__.ConvBNAct = ConvBNAct
    __main__.SqueezeExcite = SqueezeExcite
    __main__.UIB = UIB
    __main__.UIBDown = UIBDown

    if "uib" in weights_path.lower():
        register_uib_modules()

    model = YOLO(weights_path)
    return model


def detect_on_frame(model, frame_bgr, imgsz=320, conf=0.25, iou=0.45, device="cpu"):
    results = model.predict(
        source=frame_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False
    )

    result = results[0]
    annotated = frame_bgr.copy()
    num_det = 0
    det_list = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        num_det = len(boxes)

        for box, score, cls_id in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            color = get_color(int(cls_id))
            class_name = get_class_name(result, model, int(cls_id))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            draw_label(annotated, f"{class_name} {score:.2f}", x1, y1, color)

            det_list.append({
                "class_id": int(cls_id),
                "class_name": class_name,
                "confidence": float(score),
                "box": [x1, y1, x2, y2],
            })

    return annotated, num_det, det_list


def run_webcam(weights_path: str, imgsz: int, conf: float, iou: float, device: str):
    import __main__

    __main__.ConvBNAct = ConvBNAct
    __main__.SqueezeExcite = SqueezeExcite
    __main__.UIB = UIB
    __main__.UIBDown = UIBDown

    if "uib" in weights_path.lower():
        register_uib_modules()

    model = YOLO(weights_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    prev_time = time.time()
    fps_smooth = 0.0
    frame_id = 0
    last_annotated = None
    last_num_det = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        if frame_id % 3 == 0:
            annotated, num_det, _ = detect_on_frame(
                model, frame, imgsz=imgsz, conf=conf, iou=iou, device=device
            )
            last_annotated = annotated
            last_num_det = num_det
        else:
            annotated = frame.copy() if last_annotated is None else last_annotated.copy()
            num_det = last_num_det

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        fps_smooth = fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * fps

        annotated = draw_fps_panel(annotated, fps_smooth, num_det)
        cv2.imshow("Traffic Sign Detection - Webcam", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_streamlit_app():
    st.set_page_config(page_title="Traffic Sign Detection Demo", layout="wide")
    st.title("Traffic Sign Detection Demo")
    st.write("Demo nhận diện biển báo giao thông bằng YOLO11n / YOLO11n-UIB")

    with st.sidebar:
        st.header("Cấu hình")
        mode = st.selectbox("Chọn chế độ", ["Image", "Video", "Webcam"])
        
        # Model selection presets
        model_presets = {
            "YOLO11n Baseline (Production - Recommended: 70.3% mAP, 5.8ms)": "weights/yolo11n_baseline_best.pt",
            "YOLO11n-UIB-Nano + KD (Lightweight: 2.48M, 64.1% mAP)": "runs/controlled/uib_nano_kd_seed42/weights/best.pt",
            "YOLO11n-UIB Original (High-Capacity 32M, 70.9% mAP)": "weights/yolo11n_uib_best.pt",
            "Custom path...": ""
        }
        chosen_preset = st.selectbox("Chọn mô hình", list(model_presets.keys()), index=0)
        default_path = model_presets[chosen_preset] if model_presets[chosen_preset] else get_default_weights()
        weights_path = st.text_input("Đường dẫn weights", value=default_path)
        imgsz = st.selectbox("Image size", [256, 320, 416, 512, 640], index=1)
        conf = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
        iou = st.slider("IoU", 0.10, 0.95, 0.45, 0.05)
        device = st.selectbox("Device", ["cpu", "0"], index=0)

        st.markdown("---")
        st.write(f"Số class map cứng: {len(VRTSD_CLASS_NAMES)}")
        if st.checkbox("Hiện bảng class"):
            st.json(VRTSD_CLASS_NAMES)

    if not weights_path or not os.path.exists(weights_path):
        st.error("Không tìm thấy file weights.")
        st.stop()

    model = load_model(weights_path)

    if mode == "Image":
        uploaded_image = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png", "jfif"])
        if uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            t0 = time.time()
            annotated, num_det, det_list = detect_on_frame(
                model, image_bgr, imgsz=imgsz, conf=conf, iou=iou, device=device
            )
            elapsed = time.time() - t0

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Ảnh gốc")
                st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.subheader("Kết quả")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.write(f"Số detections: **{num_det}**")
            st.write(f"Thời gian xử lý: **{elapsed:.3f}s**")

            if det_list:
                st.subheader("Danh sách class nhận diện được")
                for i, det in enumerate(det_list, 1):
                    st.write(
                        f"{i}. **{det['class_name']}** | conf = {det['confidence']:.2f} | id = {det['class_id']}"
                    )
            else:
                st.info("Không phát hiện được biển báo nào.")

    elif mode == "Video":
        uploaded_video = st.file_uploader("Tải video lên", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_video is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                tmp_in.write(uploaded_video.read())
                input_video_path = tmp_in.name

            st.video(input_video_path)

            if st.button("Chạy nhận diện video"):
                cap = cv2.VideoCapture(input_video_path)
                if not cap.isOpened():
                    st.error("Không mở được video.")
                    st.stop()

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps_in = cap.get(cv2.CAP_PROP_FPS)
                if fps_in <= 0:
                    fps_in = 25.0

                out_path = str(PRED_DIR / "video_result.mp4")
                writer = cv2.VideoWriter(
                    out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps_in,
                    (width, height)
                )

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress = st.progress(0)
                frame_placeholder = st.empty()
                status_placeholder = st.empty()

                frame_idx = 0
                t_all = time.time()
                class_counter = {}

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    t0 = time.time()
                    annotated, num_det, det_list = detect_on_frame(
                        model, frame, imgsz=imgsz, conf=conf, iou=iou, device=device
                    )
                    fps_cur = 1.0 / max(time.time() - t0, 1e-6)
                    annotated = draw_fps_panel(annotated, fps_cur, num_det)
                    writer.write(annotated)

                    for det in det_list:
                        name = det["class_name"]
                        class_counter[name] = class_counter.get(name, 0) + 1

                    frame_idx += 1
                    if total_frames > 0:
                        progress.progress(min(frame_idx / total_frames, 1.0))

                    if frame_idx % 10 == 0:
                        frame_placeholder.image(
                            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            caption=f"Frame {frame_idx}",
                            use_container_width=True,
                        )

                cap.release()
                writer.release()

                elapsed = time.time() - t_all
                status_placeholder.success(f"Xong. Tổng thời gian: {elapsed:.2f}s")
                st.video(out_path)
                st.write(f"Đã lưu: `{out_path}`")

                if class_counter:
                    st.subheader("Các class xuất hiện trong video")
                    sorted_items = sorted(class_counter.items(), key=lambda x: x[1], reverse=True)
                    for name, cnt in sorted_items:
                        st.write(f"- **{name}**: {cnt} lần")
                else:
                    st.info("Không phát hiện được class nào trong video.")

    else:
        st.subheader("Webcam realtime")
        st.info("Webcam sẽ mở bằng cửa sổ OpenCV local trên chính máy của bạn.")

        if st.button("Bắt đầu nhận diện webcam"):
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "app.py"),
                "--webcam",
                "--weights", weights_path,
                "--imgsz", str(imgsz),
                "--conf", str(conf),
                "--iou", str(iou),
                "--device", str(device),
            ]
            subprocess.Popen(cmd)
            st.success("Đã mở webcam local. Nhấn Q hoặc ESC để thoát cửa sổ webcam.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--weights", type=str, default=get_default_weights())
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cpu")
    args, _ = parser.parse_known_args()

    if args.webcam:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        run_webcam(args.weights, args.imgsz, args.conf, args.iou, args.device)
    else:
        run_streamlit_app()


if __name__ == "__main__":
    main()