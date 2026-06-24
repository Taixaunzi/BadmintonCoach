"""羽毛球追踪 — HSV检测 + 卡尔曼滤波 + 轨迹/速度计算"""
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import BallConfig


@dataclass
class BallState:
    """球的状态"""
    position: Optional[np.ndarray] = None  # [x, y]
    velocity: Optional[np.ndarray] = None  # [vx, vy] px/frame
    speed: float = 0.0                     # px/frame
    speed_kmh: float = 0.0                 # km/h (估算)
    confidence: float = 0.0
    frame_idx: int = 0


class BallTracker:
    """羽毛球追踪器"""

    def __init__(self, config: BallConfig):
        self.config = config
        self._kalman: Optional[cv2.KalmanFilter] = None
        self._trajectory: deque = deque(maxlen=30)  # 最近30帧轨迹
        self._prev_state: Optional[BallState] = None
        self._lost_count = 0  # 连续丢失帧数
        self._init_kalman()

        # 像素到速度的换算系数（假设球场13.4m长，画面中占600px）
        # 1px ≈ 0.022m，30fps → 1帧=0.033s
        # speed_kmh = speed_px * 0.022 / 0.033 * 3.6 ≈ speed_px * 2.4
        self._px_to_kmh = 2.4

    def _init_kalman(self):
        """初始化卡尔曼滤波器"""
        self._kalman = cv2.KalmanFilter(4, 2)  # 4状态(x,y,vx,vy), 2观测(x,y)
        self._kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        self._kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self._kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self._kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

    def __call__(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """检测单帧球位置
        Returns:
            ball_position: [x, y] 或 None
        """
        detected = self._detect_ball(frame)

        if detected is not None:
            self._lost_count = 0
            # 更新卡尔曼滤波
            self._kalman.correct(np.array(detected, dtype=np.float32))
            predicted = self._kalman.predict()
            pos = np.array([predicted[0, 0], predicted[1, 0]])
            self._trajectory.append(pos.copy())
            return pos
        else:
            self._lost_count += 1
            # 用卡尔曼预测（最多预测10帧）
            if self._lost_count <= 10 and self._kalman is not None:
                predicted = self._kalman.predict()
                pos = np.array([predicted[0, 0], predicted[1, 0]])
                self._trajectory.append(pos.copy())
                return pos
            self._trajectory.append(None)
            return None

    def _detect_ball(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """HSV颜色检测羽毛球"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 羽毛球颜色：白色/黄色/绿色（鹅毛部分）
        masks = []

        # 白色球头
        mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
        masks.append(mask_white)

        # 黄色球头
        mask_yellow = cv2.inRange(hsv, np.array([15, 80, 180]), np.array([35, 255, 255]))
        masks.append(mask_yellow)

        # 绿色鹅毛
        mask_green = cv2.inRange(hsv, np.array([35, 40, 100]), np.array([85, 255, 255]))
        masks.append(mask_green)

        combined = masks[0]
        for m in masks[1:]:
            combined = cv2.bitwise_or(combined, m)

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        # 找轮廓
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 筛选：面积适中、接近圆形
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 5 or area > 500:
                continue

            # 圆度检查
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.3:
                continue

            # 用卡尔曼预测位置辅助筛选（如果有预测）
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # 如果有卡尔曼预测，优先选接近预测位置的
            if self._kalman is not None and self._lost_count < 5:
                pred_x = self._kalman.statePost[0, 0]
                pred_y = self._kalman.statePost[1, 0]
                dist = np.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)
                if dist > 100:  # 距离预测位置太远，跳过
                    continue
                score = circularity * 10 - dist * 0.1
            else:
                score = circularity * 10

            candidates.append((cx, cy, score, area))

        if not candidates:
            return None

        # 选得分最高的
        candidates.sort(key=lambda x: -x[2])
        best = candidates[0]
        return np.array([best[0], best[1]])

    def get_state(self, frame_idx: int) -> BallState:
        """获取当前球状态"""
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
            confidence=0.8 if self._lost_count == 0 else max(0.1, 0.8 - self._lost_count * 0.1),
            frame_idx=frame_idx,
        )

    def get_trajectory(self) -> List[Optional[np.ndarray]]:
        """获取轨迹历史"""
        return list(self._trajectory)

    def reset(self):
        self._init_kalman()
        self._trajectory.clear()
        self._prev_state = None
        self._lost_count = 0
