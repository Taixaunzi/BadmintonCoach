"""骨骼姿态估计 — 球场检测 + 运动员裁剪放大 + 姿态估计"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import PoseConfig


class CourtDetector:
    """羽毛球球场检测器"""

    def __init__(self):
        self._court_mask: Optional[np.ndarray] = None

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """检测球场区域，返回二值掩码"""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        masks = []
        # 绿色球场
        masks.append(cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255])))
        # 蓝色球场
        masks.append(cv2.inRange(hsv, np.array([90, 40, 40]), np.array([130, 255, 255])))
        # 红色球场
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
            return court_mask

        self._court_mask = combined
        return combined

    def is_in_court(self, x: float, y: float) -> bool:
        """判断点是否在球场区域内"""
        if self._court_mask is None:
            return y > 0.4
        h, w = self._court_mask.shape[:2]
        ix, iy = int(x), int(y)
        if 0 <= ix < w and 0 <= iy < h:
            r = 10
            region = self._court_mask[max(0, iy-r):min(h, iy+r), max(0, ix-r):min(w, ix+r)]
            return np.mean(region) > 30
        return False


class PoseEstimator:
    """RTMPose姿态估计器（球场检测 + 裁剪放大）"""

    def __init__(self, config: PoseConfig):
        self.config = config
        self._tracker = None
        self._detector = None
        self.court_detector = CourtDetector()

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

    def _init_detector(self):
        """初始化人体检测器（用于裁剪放大）"""
        if self._detector is not None:
            return
        try:
            from rtmlib import Body
            # 用更灵敏的检测模式
            self._detector = Body(backend=self.config.backend, device=self.config.device)
        except Exception:
            self._detector = "fallback"

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检测单帧姿态"""
        self._init_tracker()
        h, w = frame.shape[:2]

        # Step 1: 检测球场
        court_mask = self.court_detector.detect(frame)

        # Step 2: 直接用RTMPose检测（它内置了人体检测器）
        all_kpts, all_scores = self._tracker(frame)

        if len(all_kpts) == 0:
            # 没检测到人，尝试裁剪下半部分放大重试
            return self._detect_cropped(frame, court_mask)

        if len(all_kpts) == 1:
            return all_kpts, all_scores

        # Step 3: 多人时，过滤场上运动员
        player_indices = self._filter_on_court(all_kpts, all_scores, court_mask, h, w)

        if not player_indices:
            player_indices = [self._fallback_best(all_kpts, all_scores, h, w)]

        # 取前2人（支持双打）
        if len(player_indices) > 2:
            player_indices = self._top_n_by_size(all_kpts, all_scores, player_indices, 2)

        selected_kpts = np.array([all_kpts[i] for i in player_indices])
        selected_scores = np.array([all_scores[i] for i in player_indices])

        # Step 4: 如果关键点置信度太低，尝试裁剪放大重检
        mean_conf = float(np.mean(selected_scores[selected_scores > 0.3])) if np.any(selected_scores > 0.3) else 0
        if mean_conf < 0.4:
            cropped_result = self._detect_cropped(frame, court_mask)
            if cropped_result is not None:
                return cropped_result

        return selected_kpts, selected_scores

    def _detect_cropped(
        self, frame: np.ndarray, court_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """裁剪画面下半部分放大，重新检测姿态"""
        h, w = frame.shape[:2]

        # 裁剪下半部分（球场区域）
        crop_y = h // 3
        cropped = frame[crop_y:, :]
        cropped_h, cropped_w = cropped.shape[:2]

        # 如果裁剪区域太小，跳过
        if cropped_h < 100 or cropped_w < 100:
            return np.zeros((1, 17, 2)), np.zeros((1, 17))

        # 放大到至少480高
        scale = max(1.0, 480 / cropped_h)
        if scale > 1.0:
            resized = cv2.resize(cropped, (int(cropped_w * scale), int(cropped_h * scale)))
        else:
            resized = cropped

        # 在裁剪区域检测姿态
        self._init_tracker()
        kpts, scores = self._tracker(resized)

        if len(kpts) == 0:
            return np.zeros((1, 17, 2)), np.zeros((1, 17))

        # 选最大的人
        best_idx = 0
        best_area = 0
        for i in range(len(kpts)):
            sc = scores[i]
            valid = sc > 0.3
            if np.any(valid):
                vk = kpts[i][valid]
                area = (vk[:, 0].max() - vk[:, 0].min()) * (vk[:, 1].max() - vk[:, 1].min())
                if area > best_area:
                    best_area = area
                    best_idx = i

        kpt = kpts[best_idx:best_idx+1].copy()
        sc = scores[best_idx:best_idx+1]

        # 坐标映射回原图
        kpt[:, :, 0] = kpt[:, :, 0] / scale
        kpt[:, :, 1] = kpt[:, :, 1] / scale + crop_y

        return kpt, sc

    def _filter_on_court(
        self, kpts: np.ndarray, scores: np.ndarray,
        court_mask: np.ndarray, h: int, w: int
    ) -> List[int]:
        """过滤：只保留脚在球场内的人"""
        players = []
        for i in range(len(kpts)):
            kpt = kpts[i]
            sc = scores[i]

            # 找脚部（ankle: 15, 16）
            ankles = []
            for idx in [15, 16]:
                if sc[idx] > 0.3:
                    ankles.append(kpt[idx])

            if not ankles:
                valid = sc > 0.3
                if not np.any(valid):
                    continue
                vk = kpt[valid]
                lowest_y = float(vk[:, 1].max())
                center_x = float(vk[:, 0].mean())
                ankles = [(center_x, lowest_y)]

            in_court = False
            for ax, ay in ankles:
                ix, iy = int(ax), int(ay)
                if 0 <= ix < w and 0 <= iy < h:
                    r = 15
                    region = court_mask[max(0, iy-r):min(h, iy+r), max(0, ix-r):min(w, ix+r)]
                    if np.mean(region) > 30:
                        in_court = True
                        break

            if in_court:
                players.append(i)

        return players

    def _fallback_best(self, kpts: np.ndarray, scores: np.ndarray, h: int, w: int) -> int:
        """Fallback: 选最大+最靠下的人"""
        best_idx = 0
        best_score = -1
        for i in range(len(kpts)):
            sc = scores[i]
            valid = sc > 0.3
            if not np.any(valid):
                continue
            vk = kpts[i][valid]
            area = float((vk[:, 0].max() - vk[:, 0].min()) * (vk[:, 1].max() - vk[:, 1].min()))
            lowest_y = float(vk[:, 1].max()) / h
            score = area / (w * h) * 0.6 + lowest_y * 0.4
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _top_n_by_size(self, kpts, scores, indices, n):
        """按身体面积取前N人"""
        areas = []
        for idx in indices:
            sc = scores[idx]
            valid = sc > 0.3
            if np.any(valid):
                vk = kpts[idx][valid]
                area = (vk[:, 0].max() - vk[:, 0].min()) * (vk[:, 1].max() - vk[:, 1].min())
                areas.append((idx, area))
            else:
                areas.append((idx, 0))
        areas.sort(key=lambda x: -x[1])
        return [a[0] for a in areas[:n]]

    def reset(self):
        self._tracker = None
        self._detector = None
        self.court_detector = CourtDetector()
