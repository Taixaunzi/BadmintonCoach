"""球拍检测器 — 检测手腕附近是否有长条形物体（球拍）"""
from typing import List, Optional, Tuple

import cv2
import numpy as np


class RacketDetector:
    """基于手腕区域形状分析的球拍检测器"""

    def __init__(self):
        # 球拍特征：长条形，宽高比>3，面积适中
        self.min_aspect_ratio = 3.0    # 最小宽高比
        self.max_aspect_ratio = 15.0   # 最大宽高比
        self.min_length = 30           # 最小长度（像素）
        self.max_length = 200          # 最大长度（像素）
        self.search_radius = 80        # 手腕周围搜索半径（像素）

    def has_racket(self, frame: np.ndarray, keypoints: np.ndarray,
                   scores: np.ndarray) -> bool:
        """判断该人是否持有球拍
        
        检查左右手腕周围区域是否有长条形物体
        """
        h, w = frame.shape[:2]

        # 检查左右手腕
        for wrist_idx in [9, 10]:  # left_wrist, right_wrist
            if scores[wrist_idx] < 0.3:
                continue

            wx, wy = int(keypoints[wrist_idx][0]), int(keypoints[wrist_idx][1])
            if wx < 0 or wx >= w or wy < 0 or wy >= h:
                continue

            # 裁剪手腕周围区域
            r = self.search_radius
            x1, y1 = max(0, wx - r), max(0, wy - r)
            x2, y2 = min(w, wx + r), min(h, wy + r)
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            if self._detect_racket_in_roi(roi):
                return True

        return False

    def _detect_racket_in_roi(self, roi: np.ndarray) -> bool:
        """在ROI中检测球拍（长条形物体）"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)

        # 膨胀边缘，连接断裂部分
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)

        # 找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue

            # 最小外接矩形
            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect

            # 确保宽<高（长条形）
            if rw > rh:
                rw, rh = rh, rw

            if rh < 1:
                continue

            aspect_ratio = rh / rw
            length = rh

            # 检查是否符合球拍特征
            if (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio and
                    self.min_length <= length <= self.max_length):
                return True

        # 备选方案：检查是否有明亮的长条形区域（球拍线通常是白色/亮色）
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 检测亮白色区域（球拍线）
        bright = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
        bright = cv2.dilate(bright, kernel, iterations=2)

        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue

            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect
            if rw > rh:
                rw, rh = rh, rw

            if rh < 1:
                continue

            aspect_ratio = rh / rw
            if aspect_ratio > 2.5 and rh > 25:
                return True

        return False

    def detect_racket_position(
        self, frame: np.ndarray, keypoints: np.ndarray, scores: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """检测球拍中心位置（用于标注）"""
        h, w = frame.shape[:2]

        for wrist_idx in [9, 10]:
            if scores[wrist_idx] < 0.3:
                continue

            wx, wy = int(keypoints[wrist_idx][0]), int(keypoints[wrist_idx][1])
            if wx < 0 or wx >= w or wy < 0 or wy >= h:
                continue

            r = self.search_radius
            x1, y1 = max(0, wx - r), max(0, wy - r)
            x2, y2 = min(w, wx + r), min(h, wy + r)
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # 在ROI中找最亮的长条形区域
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=2)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = None
            best_score = 0

            for contour in contours:
                if cv2.contourArea(contour) < 50:
                    continue
                rect = cv2.minAreaRect(contour)
                (cx, cy), (rw, rh), angle = rect
                if rw > rh:
                    rw, rh = rh, rw
                if rh < 1:
                    continue
                ar = rh / rw
                if ar > 2.5 and rh > 25:
                    score = ar * rh
                    if score > best_score:
                        best_score = score
                        best_contour = contour

            if best_contour is not None:
                M = cv2.moments(best_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]) + x1
                    cy = int(M["m01"] / M["m00"]) + y1
                    return (cx, cy)

        return None
