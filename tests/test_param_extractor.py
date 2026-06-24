"""关节角度计算模块测试"""
import numpy as np
import pytest

from badmintoncoach.engine.param_extractor import (
    calc_body_lean,
    calc_wrist_speed,
    calculate_angle,
    extract_joint_angles,
)


class TestCalculateAngle:
    def test_right_angle(self):
        """90度角"""
        assert abs(calculate_angle([0, 0], [1, 0], [1, 1]) - 90.0) < 0.1

    def test_straight_line(self):
        """180度（直线）"""
        assert abs(calculate_angle([0, 0], [1, 0], [2, 0]) - 180.0) < 0.1

    def test_acute_angle(self):
        """锐角 ~45度"""
        angle = calculate_angle([0, 1], [0, 0], [1, 1])
        assert 40 < angle < 50

    def test_zero_length_vector(self):
        """退化情况：重合点"""
        assert calculate_angle([0, 0], [0, 0], [1, 0]) == 0.0


def _make_kpts():
    """构造一个标准站立姿态的17关键点"""
    return np.array([
        [0, 0],      # 0 nose
        [-0.5, 0.5], # 1 left_eye
        [0.5, 0.5],  # 2 right_eye
        [-1, 0],     # 3 left_ear
        [1, 0],      # 4 right_ear
        [-2, -1],    # 5 left_shoulder
        [2, -1],     # 6 right_shoulder
        [-3, -1],    # 7 left_elbow
        [3, -1],     # 8 right_elbow
        [-3, -2],    # 9 left_wrist
        [3, -2],     # 10 right_wrist
        [-1.5, -3],  # 11 left_hip
        [1.5, -3],   # 12 right_hip
        [-1.5, -5],  # 13 left_knee
        [1.5, -5],   # 14 right_knee
        [-1.5, -7],  # 15 left_ankle
        [1.5, -7],   # 16 right_ankle
    ], dtype=float)


class TestExtractJointAngles:
    def test_all_visible(self):
        kpts = _make_kpts()
        scores = np.ones(17) * 0.9
        angles = extract_joint_angles(kpts, scores)
        assert "left_elbow" in angles
        assert "right_knee" in angles
        assert len(angles) == 8

    def test_low_confidence_filtered(self):
        kpts = _make_kpts()
        scores = np.ones(17) * 0.1
        angles = extract_joint_angles(kpts, scores, threshold=0.3)
        assert len(angles) == 0

    def test_partial_visibility(self):
        kpts = _make_kpts()
        scores = np.ones(17) * 0.9
        scores[9] = 0.1  # left_wrist 低置信度
        angles = extract_joint_angles(kpts, scores)
        assert "left_elbow" not in angles  # 依赖 left_wrist
        assert "right_elbow" in angles


class TestCalcWristSpeed:
    def test_no_previous(self):
        kpts = _make_kpts()
        scores = np.ones(17) * 0.9
        assert calc_wrist_speed(kpts, scores, None) == 0.0

    def test_same_position(self):
        kpts = _make_kpts()
        scores = np.ones(17) * 0.9
        assert calc_wrist_speed(kpts, scores, kpts.copy()) == 0.0

    def test_movement(self):
        kpts = _make_kpts()
        prev = kpts.copy()
        prev[10] = [0, 0]  # right_wrist 移动
        scores = np.ones(17) * 0.9
        speed = calc_wrist_speed(kpts, scores, prev)
        assert speed > 0


class TestCalcBodyLean:
    def test_upright(self):
        kpts = _make_kpts()
        scores = np.ones(17) * 0.9
        lean = calc_body_lean(kpts, scores)
        # 图像坐标系中y轴向下，直立时脊柱向量指向下，与垂直方向[0,-1]夹角接近180°
        # 实际倾斜 = |180 - lean| 即身体偏离直立的角度
        deviation = abs(180 - lean)
        assert deviation < 20  # 基本直立（测试数据有轻微横向偏移）

    def test_low_confidence(self):
        kpts = _make_kpts()
        scores = np.ones(17) * 0.1
        assert calc_body_lean(kpts, scores) == 0.0
