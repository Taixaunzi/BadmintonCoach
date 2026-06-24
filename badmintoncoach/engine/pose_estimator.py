"""骨骼姿态估计 — 球场检测 + 运动员过滤（身高+位置+球场）"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import PoseConfig
from .racket_detector import RacketDetector


class CourtDetector:
    """羽毛球球场检测器"""

    def __init__(self):
        self._court_mask: Optional[np.ndarray] = None
        self._court_bbox: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """检测球场区域，返回二值掩码"""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        masks = []
        masks.append(cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255])))
        masks.append(cv2.inRange(hsv, np.array([90, 40, 40]), np.array([130, 255, 255])))
        m1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([15, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([165, 40, 40]), np.array([180, 255, 255]))
        masks.append(cv2.bitwise_or(m1, m2))

        combined = masks[0]
        for m in masks[1:]:
            combined = cv2.bitwise_or(combined, m)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined[:h // 3, :] = 0

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            court_mask = np.zeros_like(combined)
            cv2.fillPoly(court_mask, [largest], 255)
            self._court_mask = court_mask
            # 记录球场边界框
            x, y, bw, bh = cv2.boundingRect(largest)
            self._court_bbox = (x, y, bw, bh)
            return court_mask

        self._court_mask = combined
        self._court_bbox = (0, h // 3, w, h * 2 // 3)
        return combined


class PoseEstimator:
    """RTMPose姿态估计器（球场检测 + 运动员过滤）"""

    def __init__(self, config: PoseConfig):
        self.config = config
        self._tracker = None
        self.court_detector = CourtDetector()
        self.racket_detector = RacketDetector()

    def _init_tracker(self):
        if self._tracker is not None:
            return
        from rtmlib import Body, PoseTracker
        self._tracker = PoseTracker(
            Body,
            mode=self.config.mode,
            det_frequency=self.config.det_frequency,
            backend=self.config.backend,
            device=self.config.device,
        )

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检测单帧姿态（自动过滤场上运动员）"""
        self._init_tracker()
        h, w = frame.shape[:2]

        # Step 1: 检测球场
        court_mask = self.court_detector.detect(frame)

        # Step 2: 姿态检测
        all_kpts, all_scores = self._tracker(frame)

        if len(all_kpts) == 0:
            return np.zeros((1, 17, 2)), np.zeros((1, 17))

        if len(all_kpts) == 1:
            return all_kpts, all_scores

        # Step 3: 评分选出场上运动员
        player_indices = self._score_players(all_kpts, all_scores, court_mask, h, w, frame)

        if not player_indices:
            player_indices = [0]  # fallback

        # 只取前1人（单打模式，避免标注混乱）
        selected_kpts = all_kpts[player_indices[0]:player_indices[0]+1]
        selected_scores = all_scores[player_indices[0]:player_indices[0]+1]
        return selected_kpts, selected_scores

    def _score_players(
        self, kpts: np.ndarray, scores: np.ndarray,
        court_mask: np.ndarray, h: int, w: int, frame: np.ndarray
    ) -> List[int]:
        """对每个人评分，选出最可能是场上运动员的人

        评分维度：
        1. 球拍（最重要：拿球拍的人几乎一定是运动员）
        2. 身高占比（站着的运动员 > 坐着的观众）
        3. 身体中心在球场区域内
        4. 位置（画面中下部）
        """
        court_bbox = self.court_detector._court_bbox
        if court_bbox:
            cx, cy, cw, ch = court_bbox
            court_center_y = cy + ch / 2
            court_top = cy
            court_bottom = cy + ch
        else:
            court_center_y = h * 0.6
            court_top = h * 0.3
            court_bottom = h * 0.9

        scored = []
        for i in range(len(kpts)):
            kpt = kpts[i]
            sc = scores[i]
            valid = sc > 0.3
            if not np.any(valid):
                continue

            vk = kpt[valid]
            vs = sc[valid]

            # 身体指标
            body_cx = float(vk[:, 0].mean())
            body_cy = float(vk[:, 1].mean())
            body_height = float(vk[:, 1].max() - vk[:, 1].min())
            height_ratio = body_height / h
            body_width = float(vk[:, 0].max() - vk[:, 0].min())
            body_area = body_height * body_width
            mean_conf = float(vs.mean())

            # 0. 球拍检测（最重要）
            has_racket = self.racket_detector.has_racket(frame, kpt, sc)
            racket_score = 1.0 if has_racket else 0.0

            # 1. 身高占比分（0-1）
            height_score = min(1.0, height_ratio / 0.25)

            # 2. 球场内分（0或1）
            ix, iy = int(body_cx), int(body_cy)
            if 0 <= ix < w and 0 <= iy < h:
                in_court = court_mask[iy, ix] > 0
            else:
                in_court = False
            court_score = 1.0 if in_court else 0.0

            # 3. 位置分（0-1）
            if court_top <= body_cy <= court_bottom:
                dist_from_center = abs(body_cy - court_center_y) / (court_bottom - court_top) * 2
                position_score = max(0, 1.0 - dist_from_center)
            else:
                position_score = 0.0

            # 4. 面积分（0-1）
            area_score = min(1.0, body_area / (w * h * 0.05))

            # 综合评分：球拍权重最高
            total = (
                racket_score * 0.50 +    # 球拍最重要
                height_score * 0.20 +    # 身高
                court_score * 0.15 +     # 球场内
                position_score * 0.10 +  # 位置
                area_score * 0.05        # 面积
            )

            scored.append((i, total, height_score, court_score, position_score, area_score))

        if not scored:
            return []

        # 按总分排序
        scored.sort(key=lambda x: -x[1])

        # 返回得分>0.3的人
        result = [s[0] for s in scored if s[1] > 0.3]
        return result

    def reset(self):
        self._tracker = None
        self.court_detector = CourtDetector()
        self.racket_detector = RacketDetector()
