"""骨骼姿态估计 — 基于rtmlib的RTMPose封装"""
import numpy as np
from typing import Tuple

from ..config import PoseConfig


class PoseEstimator:
    """RTMPose姿态估计器"""

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
        """检测单帧姿态
        Args:
            frame: BGR图像 (H, W, 3)
        Returns:
            keypoints: (N, 17, 2) 关键点坐标
            scores: (N, 17) 置信度
        """
        self._init_tracker()
        keypoints, scores = self._tracker(frame)
        return keypoints, scores

    def reset(self):
        """重置追踪器状态（处理新视频时调用）"""
        self._tracker = None
