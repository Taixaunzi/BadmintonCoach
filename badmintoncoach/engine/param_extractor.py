"""运动参数提取：关节角度、速度、身体倾斜"""
import numpy as np
from typing import Dict, Optional

# COCO 17关键点索引
KPT = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

# 关节角度定义：(父关节, 当前关节, 子关节)
ANGLE_DEFS = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
}


def calculate_angle(a, b, c) -> float:
    """计算关节b处的角度（度）
    Args:
        a: [x, y] 父关节
        b: [x, y] 当前关节（角度顶点）
        c: [x, y] 子关节
    Returns:
        角度（度）
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def extract_joint_angles(
    keypoints: np.ndarray, scores: np.ndarray, threshold: float = 0.3
) -> Dict[str, float]:
    """从17个关键点提取8个主要关节角度
    Args:
        keypoints: (17, 2) 关键点坐标
        scores: (17,) 置信度
        threshold: 置信度阈值
    Returns:
        {关节名: 角度(度)} 字典
    """
    angles = {}
    for angle_name, (parent, joint, child) in ANGLE_DEFS.items():
        pi, ji, ci = KPT[parent], KPT[joint], KPT[child]
        if all(scores[i] > threshold for i in [pi, ji, ci]):
            angles[angle_name] = calculate_angle(
                keypoints[pi], keypoints[ji], keypoints[ci]
            )
    return angles


def calc_wrist_speed(
    keypoints: np.ndarray,
    scores: np.ndarray,
    prev_keypoints: Optional[np.ndarray] = None,
) -> float:
    """计算手腕最大速度（像素/帧）"""
    if prev_keypoints is None:
        return 0.0
    max_speed = 0.0
    for wrist_idx in [9, 10]:  # left_wrist, right_wrist
        if scores[wrist_idx] > 0.3:
            speed = float(np.linalg.norm(keypoints[wrist_idx] - prev_keypoints[wrist_idx]))
            max_speed = max(max_speed, speed)
    return max_speed


def calc_body_lean(keypoints: np.ndarray, scores: np.ndarray) -> float:
    """计算身体倾斜角（脊柱vs垂直方向，度）"""
    if scores[6] > 0.3 and scores[12] > 0.3:
        spine = keypoints[6] - keypoints[12]
        vertical = np.array([0, -1], dtype=float)
        norm = np.linalg.norm(spine)
        if norm < 1e-6:
            return 0.0
        cosine = np.dot(spine, vertical) / norm
        return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return 0.0
