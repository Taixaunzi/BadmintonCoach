"""球体追踪 — TrackNet热力图检测"""
from typing import Optional

import cv2
import numpy as np

from ..config import BallConfig


class BallTracker:
    """羽毛球追踪器"""

    def __init__(self, config: BallConfig):
        self.config = config
        self._model = None
        self._frame_buffer: list[np.ndarray] = []

    def __call__(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """检测单帧球位置
        Args:
            frame: BGR图像 (H, W, 3)
        Returns:
            ball_position: [x, y] 或 None（未检测到）
        """
        resized = cv2.resize(frame, (640, 360))
        self._frame_buffer.append(resized)
        if len(self._frame_buffer) > 3:
            self._frame_buffer.pop(0)

        if len(self._frame_buffer) < 3:
            return None

        # TODO: 集成真实TrackNet ONNX模型
        # 目前用颜色检测作为占位实现
        return self._detect_by_color(frame)

    def _detect_by_color(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """基于颜色的简单球体检测（占位实现）"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 羽毛球通常是白色/黄色
        lower = np.array([15, 50, 200])
        upper = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 20:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return np.array([cx, cy], dtype=float)
        return None

    def reset(self):
        self._frame_buffer = []
