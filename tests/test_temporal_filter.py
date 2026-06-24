"""测试第一层时序过滤器"""
import pytest
import numpy as np

from badmintoncoach.engine.temporal_filter import (
    SlidingWindow,
    ActivityStateMachine,
    ActivityState,
    SwingDetector,
    AngleAnomalyDetector,
    TemporalConfig,
    TemporalFilter,
)
from badmintoncoach.models.schemas import FrameData, JointAngles


def make_frame(idx: int, wrist_speed: float = 0.0, body_lean: float = 0.0,
               left_elbow: float = 120.0, right_elbow: float = 120.0) -> FrameData:
    """构造测试帧"""
    return FrameData(
        frame_idx=idx,
        timestamp=idx / 30.0,
        keypoints=[[0.0, 0.0]] * 17,
        scores=[0.9] * 17,
        joint_angles=JointAngles(
            left_elbow=left_elbow,
            right_elbow=right_elbow,
            left_knee=140.0,
            right_knee=140.0,
            left_shoulder=60.0,
            right_shoulder=60.0,
            left_hip=170.0,
            right_hip=170.0,
        ),
        wrist_speed=wrist_speed,
        body_lean=body_lean,
    )


# ============================================================
# 滑窗聚合
# ============================================================
class TestSlidingWindow:
    def test_basic_stats(self):
        w = SlidingWindow(size=5)
        for v in [10, 20, 30, 40, 50]:
            w.push(v)
        s = w.stats()
        assert s.mean == 30.0
        assert s.peak == 50.0
        assert s.valley == 10.0
        assert s.range == 40.0
        assert s.count == 5

    def test_not_full(self):
        w = SlidingWindow(size=10)
        w.push(5)
        w.push(15)
        assert not w.full
        assert w.stats().count == 2

    def test_eviction(self):
        w = SlidingWindow(size=3)
        for v in [1, 2, 3, 4, 5]:
            w.push(v)
        s = w.stats()
        assert s.count == 3
        assert s.mean == 4.0  # [3, 4, 5]


# ============================================================
# 活动状态机
# ============================================================
class TestActivityStateMachine:
    def test_idle_to_playing(self):
        sm = ActivityStateMachine(
            playing_threshold=800, moving_threshold=200, cooldown_frames=3
        )
        assert sm.state == ActivityState.IDLE
        sm.update(100)
        assert sm.state == ActivityState.IDLE
        sm.update(500)
        assert sm.state == ActivityState.MOVING
        sm.update(1000)
        assert sm.state == ActivityState.PLAYING

    def test_cooldown(self):
        sm = ActivityStateMachine(
            playing_threshold=800, moving_threshold=200, cooldown_frames=3
        )
        sm.update(1000)
        assert sm.state == ActivityState.PLAYING
        # 速度骤降
        sm.update(50)
        assert sm.state == ActivityState.COOLDOWN
        sm.update(50)
        assert sm.state == ActivityState.COOLDOWN
        sm.update(50)
        assert sm.state == ActivityState.COOLDOWN
        sm.update(50)
        assert sm.state == ActivityState.IDLE

    def test_is_active(self):
        sm = ActivityStateMachine(
            playing_threshold=800, moving_threshold=200, cooldown_frames=2
        )
        sm.update(1000)
        assert sm.is_active
        sm.update(50)
        assert sm.is_active  # cooldown
        sm.update(50)
        assert sm.is_active  # cooldown
        sm.update(50)
        assert not sm.is_active  # idle


# ============================================================
# 挥拍检测
# ============================================================
class TestSwingDetector:
    def test_detect_swing_peak(self):
        sd = SwingDetector(
            speed_threshold=500, accel_threshold=200, min_gap_frames=5
        )
        # 模拟速度曲线：低→上升→峰值→下降
        speeds = [100, 200, 400, 700, 1000, 1200, 900, 500, 200]
        swing = None
        for i, s in enumerate(speeds):
            result = sd.update(i, s)
            if result:
                swing = result
        assert swing is not None
        assert swing.peak_speed == 1200
        assert swing.peak_frame == 5

    def test_no_swing_below_threshold(self):
        sd = SwingDetector(speed_threshold=500, accel_threshold=200, min_gap_frames=5)
        speeds = [100, 150, 200, 250, 200, 150]
        swing = None
        for i, s in enumerate(speeds):
            result = sd.update(i, s)
            if result:
                swing = result
        assert swing is None

    def test_min_gap(self):
        sd = SwingDetector(speed_threshold=500, accel_threshold=200, min_gap_frames=10)
        # 第一次挥拍: frame 3 峰值
        speeds1 = [100, 400, 800, 1200, 800, 300]
        # 第二次挥拍: frame 9 峰值（距 frame 3 = 6 < min_gap=10，应被忽略）
        speeds2 = [100, 400, 800, 1100, 800, 300]
        swings = []
        for i, s in enumerate(speeds1):
            result = sd.update(i, s)
            if result:
                swings.append(result)
        for i, s in enumerate(speeds2, start=6):
            result = sd.update(i, s)
            if result:
                swings.append(result)
        assert len(swings) == 1, f"间隔不足应忽略第二次，实际: {len(swings)}"


# ============================================================
# 角度异常检测
# ============================================================
class TestAngleAnomalyDetector:
    def test_short_spike_ignored(self):
        ad = AngleAnomalyDetector(min_duration=5)
        # 3帧异常，然后恢复 → 不应检测到
        result = None
        for i in range(3):
            result = ad.update(i, "left_elbow", 50.0, 90.0, 170.0)
        # 第4帧恢复
        result = ad.update(3, "left_elbow", 120.0, 90.0, 170.0)
        assert result is None

    def test_sustained_anomaly_detected(self):
        ad = AngleAnomalyDetector(min_duration=5)
        # 6帧持续异常
        result = None
        for i in range(6):
            result = ad.update(i, "left_elbow", 50.0, 90.0, 170.0)
        # 第7帧恢复
        result = ad.update(6, "left_elbow", 120.0, 90.0, 170.0)
        assert result is not None
        assert result.joint == "left_elbow"
        assert result.direction == "too_closed"
        assert result.start_frame == 0
        assert result.end_frame == 6

    def test_flush(self):
        ad = AngleAnomalyDetector(min_duration=5)
        for i in range(7):
            ad.update(i, "right_knee", 60.0, 100.0, 170.0)
        results = ad.flush(7)
        assert len(results) == 1
        assert results[0].joint == "right_knee"


# ============================================================
# 集成测试：TemporalFilter
# ============================================================
class TestTemporalFilter:
    def test_full_pipeline(self):
        cfg = TemporalConfig(
            window_size=5,
            playing_speed_threshold=600,
            moving_speed_threshold=200,
            cooldown_frames=3,
            swing_speed_threshold=400,
            swing_accel_threshold=200,
            swing_min_gap=5,
            angle_min_duration=3,
        )
        tf = TemporalFilter(cfg)
        angle_ranges = {"left_elbow": [90, 170], "right_elbow": [90, 170]}

        # 模拟一段打球序列（含尖锐速度峰值）
        frames = []
        for i in range(40):
            if i < 5:
                speed = 50       # 休息
                elbow = 120.0
            elif i < 10:
                speed = 300      # 走位
                elbow = 120.0
            elif i == 10:
                speed = 300      # 准备
                elbow = 120.0
            elif i == 11:
                speed = 600      # 加速
                elbow = 120.0
            elif i == 12:
                speed = 1200     # 尖锐峰值！accel=600
                elbow = 120.0
            elif i == 13:
                speed = 900      # 减速
                elbow = 70.0     # 肘关节角度异常开始
            elif i == 14:
                speed = 500
                elbow = 65.0
            elif i < 20:
                speed = 200
                elbow = 60.0     # 持续异常
            else:
                speed = 100      # 恢复
                elbow = 120.0
            frames.append(make_frame(i, wrist_speed=speed, left_elbow=elbow))

        all_results = []
        for fd in frames:
            result = tf.process_frame(fd, angle_ranges)
            all_results.append(result)

        # 至少应该有一次挥拍检测
        swings = [r.swing for r in all_results if r.swing is not None]
        assert len(swings) >= 1, f"应检测到挥拍，实际: {len(swings)}"

        # 活动状态应有变化
        states = [r.activity_state for r in all_results]
        assert ActivityState.PLAYING in states, "应出现PLAYING状态"
        assert ActivityState.IDLE in states, "应出现IDLE状态"
