"""羽毛球追踪 — TrackNet + GPU加速 + ONNX推理"""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ..config import BallConfig


@dataclass
class BallState:
    position: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    speed: float = 0.0
    speed_kmh: float = 0.0
    confidence: float = 0.0
    frame_idx: int = 0


class BallTracker:
    """基于TrackNet的羽毛球追踪器（支持GPU/ONNX加速）"""

    def __init__(self, config: BallConfig):
        self.config = config
        self._model = None
        self._onnx_session = None
        self._trajectory: deque = deque(maxlen=30)
        self._frame_buffer: list = []
        self._lost_count = 0
        self._px_to_kmh = 2.4
        self._width = 640
        self._height = 360
        self._device = None
        self._backend = None  # 'pytorch' or 'onnx'

    def _resolve_device(self) -> str:
        """解析设备选择"""
        device = self.config.device
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return device

    def _init_model(self):
        """加载模型（优先ONNX，回退PyTorch）"""
        if self._model is not None or self._onnx_session is not None:
            return

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            model_path = Path(__file__).parent.parent.parent / "models" / "tracknet_best.pth"

        if not model_path.exists():
            raise FileNotFoundError(
                f"TrackNet模型未找到: {model_path}\n"
                "请下载: https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"
            )

        self._device = self._resolve_device()

        # 尝试ONNX加速
        onnx_path = model_path.with_suffix(".onnx")
        if onnx_path.exists():
            self._load_onnx(onnx_path)
        else:
            self._load_pytorch(model_path)

    def _load_pytorch(self, model_path: Path):
        """加载PyTorch模型"""
        import torch
        from .tracknet_model import BallTrackerNet

        self._model = BallTrackerNet()
        self._model.load_state_dict(
            torch.load(str(model_path), map_location=self._device, weights_only=True)
        )
        self._model.to(self._device)
        self._model.eval()
        self._backend = "pytorch"

    def _load_onnx(self, onnx_path: Path):
        """加载ONNX模型（更快）"""
        try:
            import onnxruntime as ort

            providers = []
            if self._device == "cuda":
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            self._onnx_session = ort.InferenceSession(
                str(onnx_path), providers=providers
            )
            self._backend = "onnx"
        except ImportError:
            # onnxruntime未安装，回退PyTorch
            self._load_pytorch(onnx_path.with_suffix(".pth"))

    def export_onnx(self, output_path: Optional[str] = None) -> str:
        """导出ONNX模型（一次性操作，后续推理用ONNX加速）"""
        import torch
        from .tracknet_model import BallTrackerNet

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            model_path = Path(__file__).parent.parent.parent / "models" / "tracknet_best.pth"

        model = BallTrackerNet()
        model.load_state_dict(
            torch.load(str(model_path), map_location="cpu", weights_only=True)
        )
        model.eval()

        if output_path is None:
            output_path = str(model_path.with_suffix(".onnx"))

        dummy = torch.randn(1, 9, 360, 640)
        torch.onnx.export(
            model, dummy, output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=11,
        )
        return output_path

    def __call__(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """检测单帧球位置"""
        self._init_model()

        resized = cv2.resize(frame, (self._width, self._height))
        self._frame_buffer.append(resized)
        if len(self._frame_buffer) < 3:
            return None
        if len(self._frame_buffer) > 3:
            self._frame_buffer = self._frame_buffer[-3:]

        detected = self._detect()

        if detected is not None:
            self._lost_count = 0
            h_orig, w_orig = frame.shape[:2]
            x = detected[0] * w_orig / self._width
            y = detected[1] * h_orig / self._height
            pos = np.array([x, y])
            self._trajectory.append(pos.copy())
            return pos
        else:
            self._lost_count += 1
            self._trajectory.append(None)
            return None

    def _detect(self) -> Optional[tuple]:
        """推理"""
        frames = self._frame_buffer[-3:]
        imgs = np.concatenate(frames, axis=2).astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        if self._backend == "onnx" and self._onnx_session is not None:
            output = self._onnx_session.run(None, {"input": inp})[0]
            output = output.argmax(axis=1)
        else:
            import torch
            with torch.no_grad():
                out = self._model(torch.from_numpy(inp).float().to(self._device))
                output = out.argmax(dim=1).detach().cpu().numpy()

        return self._postprocess(output)

    @staticmethod
    def _postprocess(feature_map, scale=2):
        """热力图后处理"""
        feature_map = feature_map.astype(np.float32)
        feature_map *= 255
        feature_map = feature_map.reshape((360, 640))
        feature_map = feature_map.astype(np.uint8)
        _, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(
            heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
            param1=50, param2=2, minRadius=2, maxRadius=7
        )
        if circles is not None and len(circles) >= 1:
            return (circles[0][0][0] * scale, circles[0][0][1] * scale)
        return None

    def get_state(self, frame_idx: int) -> BallState:
        pos = self._trajectory[-1] if self._trajectory else None
        speed = 0.0
        velocity = None
        if pos is not None and len(self._trajectory) >= 2:
            prev_pos = self._trajectory[-2]
            if prev_pos is not None:
                velocity = pos - prev_pos
                speed = float(np.linalg.norm(velocity))
        return BallState(
            position=pos, velocity=velocity, speed=speed,
            speed_kmh=speed * self._px_to_kmh,
            confidence=0.9 if self._lost_count == 0 else max(0.1, 0.9 - self._lost_count * 0.1),
            frame_idx=frame_idx,
        )

    def get_trajectory(self) -> List[Optional[np.ndarray]]:
        return list(self._trajectory)

    def get_info(self) -> dict:
        """返回当前推理后端信息"""
        return {
            "backend": self._backend or "not_loaded",
            "device": self._device or "not_loaded",
        }

    def reset(self):
        self._trajectory.clear()
        self._frame_buffer.clear()
        self._lost_count = 0
