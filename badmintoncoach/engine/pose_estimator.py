"""骨骼姿态估计 — 基于rtmlib的RTMPose封装（含运动员过滤）"""
from typing import Optional, Tuple

import numpy as np

from ..config import PoseConfig


class PoseEstimator:
    """RTMPose姿态估计器（含运动员过滤）"""

    def __init__(self, config: PoseConfig):
        self.config = config
        self._tracker = None

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
        all_kpts, all_scores = self._tracker(frame)

        if len(all_kpts) == 0:
            return np.zeros((1, 17, 2)), np.zeros((1, 17))

        if len(all_kpts) == 1:
            return all_kpts, all_scores

        # 多人时，过滤出场上运动员
        h, w = frame.shape[:2]
        best_idx = self._filter_player(all_kpts, all_scores, w, h)
        return all_kpts[best_idx:best_idx+1], all_scores[best_idx:best_idx+1]

    def _filter_player(
        self, kpts: np.ndarray, scores: np.ndarray, img_w: int, img_h: int
    ) -> int:
        """从多人中选出场上运动员

        策略（针对羽毛球比赛直播画面）：
        1. 排除画面顶部1/3的人（观众席/裁判）
        2. 在剩余的人中，选择身体面积最大的（场上球员比场外人大）
        3. 同等大小时，优先选更接近画面中央的
        """
        best_idx = 0
        best_score = -1.0

        for i in range(len(kpts)):
            kpt = kpts[i]
            sc = scores[i]

            valid_mask = sc > 0.3
            if not np.any(valid_mask):
                continue

            valid_kpts = kpt[valid_mask]
            valid_scores = sc[valid_mask]

            # 人体中心位置
            body_cx = float(np.mean(valid_kpts[:, 0]))
            body_cy = float(np.mean(valid_kpts[:, 1]))

            # 身体面积估算（关键点包围盒面积）
            x_min, y_min = valid_kpts.min(axis=0)
            x_max, y_max = valid_kpts.max(axis=0)
            body_area = float((x_max - x_min) * (y_max - y_min))

            # 画面高度占比（越大说明人越近/越大）
            height_ratio = (y_max - y_min) / img_h

            # === 过滤规则 ===

            # 规则1: 排除画面顶部1/3（观众席）
            if body_cy < img_h * 0.33:
                continue

            # 规则2: 排除太小的人（可能是远景观众）
            if height_ratio < 0.15:
                continue

            # === 评分 ===

            # 面积分（越大越好）
            area_score = min(1.0, body_area / (img_w * img_h * 0.1))

            # 位置分：中下部最佳
            y_pos_score = body_cy / img_h  # 越靠下分越高
            x_dist = abs(body_cx - img_w / 2) / (img_w / 2)
            x_pos_score = 1.0 - x_dist * 0.3  # 越接近中央分越高

            # 置信度分
            conf_score = float(np.mean(valid_scores))

            # 综合评分：面积权重最大
            total = area_score * 0.5 + y_pos_score * 0.2 + x_pos_score * 0.1 + conf_score * 0.2

            if total > best_score:
                best_score = total
                best_idx = i

        return best_idx

    def reset(self):
        """重置追踪器状态"""
        self._tracker = None
