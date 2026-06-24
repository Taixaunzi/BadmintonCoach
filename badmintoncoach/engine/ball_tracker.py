"""羽毛球追踪 — 帧差法运动检测 + 颜色筛选 + 卡尔曼滤波"""
from collections import deque
from dataclasses import dataclass
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
    """羽毛球追踪器"""

    def __init__(self, config: BallConfig):
        self.config = config
        self._kalman: Optional[cv2.KalmanFilter] = None
        self._trajectory: deque = deque(maxlen=30)
        self._prev_gray: Optional[np.ndarray] = None
        self._lost_count = 0
        self._init_kalman()
        self._px_to_kmh = 2.4

    def _init_kalman(self):
        self._kalman = cv2.KalmanFilter(4, 2)
        self._kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self._kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self._kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self._kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

    def __call__(self, frame: np.ndarray) -> Optional[np.ndarray]:
        detected = self._detect_ball(frame)

        if detected is not None:
            self._lost_count = 0
            self._kalman.correct(np.array(detected, dtype=np.float32))
            self._kalman.predict()
            self._trajectory.append(detected.copy())
            return detected
        else:
            self._lost_count += 1
            # 不用预测，直接返回None（只信真实检测）
            self._trajectory.append(None)
            return None

    def _detect_ball(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """帧差法检测运动物体 + 颜色筛选羽毛球"""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 1: 帧差法 — 只保留运动区域
        if self._prev_gray is not None:
            diff = cv2.absdiff(self._prev_gray, gray)
            _, motion_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            # 膨胀运动区域
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
        else:
            motion_mask = np.ones((h, w), dtype=np.uint8) * 255
        self._prev_gray = gray.copy()

        # Step 2: 颜色检测 — 羽毛球颜色（白色/黄色）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 白色（球头）
        mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 40, 255]))
        # 黄色（球头）
        mask_yellow = cv2.inRange(hsv, np.array([15, 80, 180]), np.array([35, 255, 255]))
        color_mask = cv2.bitwise_or(mask_white, mask_yellow)

        # Step 3: 合并 — 运动区域 AND 颜色匹配
        combined = cv2.bitwise_and(motion_mask, color_mask)

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        # Step 4: 找轮廓
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Step 5: 筛选
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            # 羽毛球在画面中通常很小
            if area < 2 or area > 300:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # 圆度
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # 如果有卡尔曼预测，优先选接近预测位置的
            if self._kalman is not None and self._lost_count < 5:
                pred_x = self._kalman.statePost[0, 0]
                pred_y = self._kalman.statePost[1, 0]
                dist = np.sqrt((cx - pred_x) ** 2 + (cy - pred_y) ** 2)
                if dist > 150:
                    continue
                score = circularity * 10 + area * 0.1 - dist * 0.05
            else:
                score = circularity * 10 + area * 0.1

            candidates.append((cx, cy, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[2])
        return np.array([candidates[0][0], candidates[0][1]])

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
            confidence=0.8 if self._lost_count == 0 else max(0.1, 0.8 - self._lost_count * 0.1),
            frame_idx=frame_idx,
        )

    def get_trajectory(self) -> List[Optional[np.ndarray]]:
        return list(self._trajectory)

    def reset(self):
        self._init_kalman()
        self._trajectory.clear()
        self._prev_gray = None
        self._lost_count = 0
