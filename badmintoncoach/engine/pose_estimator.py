"""骨骼姿态估计 — 球场检测 + 运动员过滤"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import PoseConfig


class CourtDetector:
    """羽毛球球场检测器"""

    def __init__(self):
        self._court_mask: Optional[np.ndarray] = None
        self._initialized = False

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """检测球场区域，返回二值掩码（255=球场，0=非球场）
        
        羽毛球场特征：
        - 绿色/蓝色/红色地面
        - 白色线条
        - 占据画面中下部
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 检测多种球场颜色
        masks = []

        # 绿色球场（常见）
        mask_green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        masks.append(mask_green)

        # 蓝色球场（国际比赛常见）
        mask_blue = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([130, 255, 255]))
        masks.append(mask_blue)

        # 红色/橙色球场
        mask_red = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([15, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([165, 40, 40]), np.array([180, 255, 255]))
        masks.append(cv2.bitwise_or(mask_red, mask_red2))

        # 合并所有颜色掩码
        combined = masks[0]
        for m in masks[1:]:
            combined = cv2.bitwise_or(combined, m)

        # 形态学操作：去除噪点，填充空洞
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        # 只保留画面下半部分（球场通常在下半部分）
        combined[:h // 3, :] = 0

        # 找最大连通域（球场）
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            court_mask = np.zeros_like(combined)
            cv2.fillPoly(court_mask, [largest], 255)
            self._court_mask = court_mask
            self._initialized = True
            return court_mask

        self._initialized = True
        return combined

    def is_in_court(self, x: float, y: float, margin: float = 0.1) -> bool:
        """判断点是否在球场区域内（带边距容差）"""
        if self._court_mask is None:
            # 未检测到球场，用简单规则：y > 画面高度的40%
            return y > 0.4

        h, w = self._court_mask.shape[:2]
        ix, iy = int(x), int(y)
        if 0 <= ix < w and 0 <= iy < h:
            # 检查该点及周围区域
            r = max(5, int(min(w, h) * margin))
            region = self._court_mask[max(0, iy-r):min(h, iy+r), max(0, ix-r):min(w, ix+r)]
            return np.mean(region) > 50
        return False


class PoseEstimator:
    """RTMPose姿态估计器（含球场检测+运动员过滤）"""

    def __init__(self, config: PoseConfig):
        self.config = config
        self._tracker = None
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

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检测单帧姿态（自动过滤场上运动员）"""
        self._init_tracker()

        # Step 1: 检测球场
        court_mask = self.court_detector.detect(frame)

        # Step 2: 姿态检测
        all_kpts, all_scores = self._tracker(frame)

        if len(all_kpts) == 0:
            return np.zeros((1, 17, 2)), np.zeros((1, 17))

        if len(all_kpts) == 1:
            return all_kpts, all_scores

        # Step 3: 过滤 - 只保留脚在球场区域内的人
        player_indices = self._filter_on_court(all_kpts, all_scores, court_mask)

        if not player_indices:
            # 没找到场上球员，fallback到最大+最低的人
            player_indices = [self._fallback_largest_lowest(all_kpts, all_scores, frame.shape)]

        # 如果场上有多人（双打），取前2人；否则取1人
        if len(player_indices) > 2:
            # 按身体面积排序，取最大的2人
            areas = []
            for idx in player_indices:
                kpt = all_kpts[idx]
                sc = all_scores[idx]
                valid = sc > 0.3
                if np.any(valid):
                    vk = kpt[valid]
                    area = (vk[:, 0].max() - vk[:, 0].min()) * (vk[:, 1].max() - vk[:, 1].min())
                    areas.append((idx, area))
                else:
                    areas.append((idx, 0))
            areas.sort(key=lambda x: -x[1])
            player_indices = [a[0] for a in areas[:2]]

        # 返回选中的运动员
        selected_kpts = np.array([all_kpts[i] for i in player_indices])
        selected_scores = np.array([all_scores[i] for i in player_indices])
        return selected_kpts, selected_scores

    def _filter_on_court(
        self, kpts: np.ndarray, scores: np.ndarray, court_mask: np.ndarray
    ) -> List[int]:
        """过滤：只保留脚部在球场区域内的人"""
        players = []
        h, w = court_mask.shape[:2]

        for i in range(len(kpts)):
            kpt = kpts[i]
            sc = scores[i]

            # 找到脚部关键点（ankle: 15, 16）
            ankles = []
            for ankle_idx in [15, 16]:
                if sc[ankle_idx] > 0.3:
                    ankles.append(kpt[ankle_idx])

            if not ankles:
                # 没有可见的脚，用身体最低点
                valid_mask = sc > 0.3
                if not np.any(valid_mask):
                    continue
                valid_kpts = kpt[valid_mask]
                lowest_y = float(valid_kpts[:, 1].max())
                center_x = float(valid_kpts[:, 0].mean())
                ankles = [(center_x, lowest_y)]

            # 检查脚是否在球场内
            in_court_count = 0
            for ax, ay in ankles:
                ix, iy = int(ax), int(ay)
                if 0 <= ix < w and 0 <= iy < h:
                    # 检查脚周围区域
                    r = 10
                    region = court_mask[max(0, iy-r):min(h, iy+r), max(0, ix-r):min(w, ix+r)]
                    if np.mean(region) > 30:
                        in_court_count += 1

            if in_court_count > 0:
                players.append(i)

        return players

    def _fallback_largest_lowest(
        self, kpts: np.ndarray, scores: np.ndarray, frame_shape: tuple
    ) -> int:
        """Fallback: 选择身体最大且最靠下的人"""
        h, w = frame_shape[:2]
        best_idx = 0
        best_score = -1

        for i in range(len(kpts)):
            kpt = kpts[i]
            sc = scores[i]
            valid = sc > 0.3
            if not np.any(valid):
                continue

            vk = kpt[valid]
            area = float((vk[:, 0].max() - vk[:, 0].min()) * (vk[:, 1].max() - vk[:, 1].min()))
            lowest_y = float(vk[:, 1].max()) / h

            score = area / (w * h) * 0.6 + lowest_y * 0.4
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def reset(self):
        """重置追踪器状态"""
        self._tracker = None
        self.court_detector = CourtDetector()
