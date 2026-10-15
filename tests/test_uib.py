import sys
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import block implementations from app
from app import UIB, UIBDown, ConvBNAct, SqueezeExcite, register_uib_modules


class TestUIBBlock(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_uib_stride1_same_channels_residual(self):
        """Test UIB with stride=1 and c1==c2 enables residual add and keeps spatial size."""
        b, c, h, w = 2, 64, 32, 32
        x = torch.randn(b, c, h, w, device=self.device)
        block = UIB(c1=64, c2=64, k=3, s=1, e=6.0).to(self.device)

        self.assertTrue(block.use_res, "Residual connection should be True when s=1 and c1==c2")
        y = block(x)
        self.assertEqual(y.shape, (b, c, h, w))

    def test_uib_stride1_diff_channels_no_residual(self):
        """Test UIB with stride=1 and c1!=c2 disables residual addition to prevent shape mismatch."""
        b, h, w = 2, 32, 32
        x = torch.randn(b, 32, h, w, device=self.device)
        block = UIB(c1=32, c2=64, k=3, s=1, e=6.0).to(self.device)

        self.assertFalse(block.use_res, "Residual connection should be False when c1 != c2")
        y = block(x)
        self.assertEqual(y.shape, (b, 64, h, w))

    def test_uib_down_stride2(self):
        """Test UIBDown halves spatial resolution (stride=2)."""
        b, c_in, c_out, h, w = 2, 32, 64, 32, 32
        x = torch.randn(b, c_in, h, w, device=self.device)
        block = UIBDown(c1=c_in, c2=c_out, k=3, s=2, e=6.0).to(self.device)

        y = block(x)
        self.assertEqual(y.shape, (b, c_out, h // 2, w // 2))

    def test_squeeze_excite_block(self):
        """Test SqueezeExcite recalibration preserves tensor shape."""
        b, c, h, w = 2, 64, 16, 16
        x = torch.randn(b, c, h, w, device=self.device)
        se = SqueezeExcite(c=64, r=0.25).to(self.device)
        y = se(x)
        self.assertEqual(y.shape, (b, c, h, w))

    def test_load_trained_uib_checkpoint_weights_dir(self):
        """Test loading trained YOLO11n-UIB weights from weights/ directory."""
        import __main__
        __main__.UIB = UIB
        __main__.UIBDown = UIBDown
        __main__.ConvBNAct = ConvBNAct
        __main__.SqueezeExcite = SqueezeExcite
        register_uib_modules()

        from ultralytics import YOLO
        weights_path = PROJECT_ROOT / "weights" / "yolo11n_uib_best.pt"
        self.assertTrue(weights_path.exists(), f"Weights file missing: {weights_path}")

        model = YOLO(str(weights_path))
        model.model.to(self.device)
        model.model.eval()

        dummy_input = torch.randn(1, 3, 640, 640, device=self.device)
        with torch.no_grad():
            out = model.model(dummy_input)

        out_tensor = out[0] if isinstance(out, (list, tuple)) else out
        self.assertEqual(out_tensor.shape[0], 1)
        self.assertEqual(out_tensor.shape[1], 59 + 4, f"Output channels should be 63, got {out_tensor.shape[1]}")

    def test_load_trained_baseline_checkpoint_weights_dir(self):
        """Test loading trained baseline YOLO11n weights from weights/ directory."""
        from ultralytics import YOLO
        baseline_path = PROJECT_ROOT / "weights" / "yolo11n_baseline_best.pt"
        self.assertTrue(baseline_path.exists(), f"Baseline weights missing: {baseline_path}")

        model = YOLO(str(baseline_path))
        model.model.to(self.device)
        model.model.eval()

        dummy_input = torch.randn(1, 3, 640, 640, device=self.device)
        with torch.no_grad():
            out = model.model(dummy_input)

        out_tensor = out[0] if isinstance(out, (list, tuple)) else out
        self.assertEqual(out_tensor.shape[0], 1)
        self.assertEqual(out_tensor.shape[1], 59 + 4)

    def test_end_to_end_predict_nms(self):
        """Test complete Ultralytics predict() inference pipeline including preprocessing and NMS."""
        import __main__
        __main__.UIB = UIB
        __main__.UIBDown = UIBDown
        __main__.ConvBNAct = ConvBNAct
        __main__.SqueezeExcite = SqueezeExcite
        register_uib_modules()

        from ultralytics import YOLO
        weights_path = PROJECT_ROOT / "weights" / "yolo11n_uib_best.pt"
        model = YOLO(str(weights_path))

        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model.predict(source=dummy_img, imgsz=640, conf=0.25, iou=0.45, verbose=False)

        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertTrue(hasattr(res, "boxes"))
        self.assertTrue(hasattr(res, "speed"))
        self.assertIn("preprocess", res.speed)
        self.assertIn("inference", res.speed)
        self.assertIn("postprocess", res.speed)


if __name__ == "__main__":
    unittest.main()
