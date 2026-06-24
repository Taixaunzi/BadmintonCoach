"""羽毛球追踪 — TrackNet深度学习模型 + 卡尔曼滤波"""
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
    """基于TrackNet的羽毛球追踪器"""

    def __init__(self, config: BallConfig):
        self.config = config
        self._model = None
        self._trajectory: deque = deque(maxlen=30)
        self._frame_buffer: list = []  # 缓存3帧
        self._lost_count = 0
        self._px_to_kmh = 2.4  # 像素速度→km/h换算系数
        self._width = 640
        self._height = 360

    def _init_model(self):
        """加载TrackNet模型"""
        if self._model is not None:
            return

        import torch
        from .tracknet_model import BallTrackerNet

        model_path = self.config.model_path
        if not Path(model_path).exists():
            # 尝试默认路径
            model_path = str(Path(__file__).parent.parent.parent / "models" / "tracknet_best.pth")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"TrackNet模型未找到: {model_path}\n"
                "请下载: https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"
            )

        self._model = BallTrackerNet()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.load_state_dict(torch.load(model_path, map_location=device))
        self._model.to(device)
        self._model.eval()
        self._device = device

    def __call__(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """检测单帧球位置"""
        self._init_model()

        # 缩放到TrackNet输入尺寸
        resized = cv2.resize(frame, (self._width, self._height))
        self._frame_buffer.append(resized)

        # 需要至少3帧
        if len(self._frame_buffer) < 3:
            return None
        # 只保留最近3帧
        if len(self._frame_buffer) > 3:
            self._frame_buffer = self._frame_buffer[-3:]

        # TrackNet推理
        detected = self._detect_with_tracknet()

        if detected is not None:
            self._lost_count = 0
            # 坐标映射回原图尺寸
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

    def _detect_with_tracknet(self) -> Optional[tuple]:
        """用TrackNet模型检测球位置"""
        import torch

        # 准备输入：3帧拼接为9通道
        frames = self._frame_buffer[-3:]
        imgs = np.concatenate(frames, axis=2)  # (360, 640, 9)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)  # (9, 360, 640)
        inp = np.expand_dims(imgs, axis=0)  # (1, 9, 360, 640)

        with torch.no_grad():
            out = self._model(torch.from_numpy(inp).float().to(self._device))

        # 后处理：argmax → 热力图 → 球位置
        output = out.argmax(dim=1).detach().cpu().numpy()
        x, y = self._postprocess(output)

        if x is not None and y is not None:
            return (x, y)
        return None

    @staticmethod
    def _postprocess(feature_map, scale=2):
        """热力图后处理：阈值+HoughCircles"""
        feature_map = feature_map.astype(np.float32)
        feature_map *= 255
        feature_map = feature_map.reshape((360, 640))
        feature_map = feature_map.astype(np.uint8)
        _, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(
            heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
            param1=50, param2=2, minRadius=2, maxRadius=7
        )
        x, y = None, None
        if circles is not None:
            if len(circles) >= 1:
                x = circles[0][0][0] * scale
                y = circles[0][0][1] * scale
        return x, y

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
            position=pos,
            velocity=velocity,
            speed=speed,
            speed_kmh=speed * self._px_to_kmh,
            confidence=0.9 if self._lost_count == 0 else max(0.1, 0.9 - self._lost_count * 0.1),
            frame_idx=frame_idx,
        )

    def get_trajectory(self) -> List[Optional[np.ndarray]]:
        return list(self._trajectory)

    def reset(self):
        self._trajectory.clear()
        self._frame_buffer.clear()
        self._lost_count = 0
