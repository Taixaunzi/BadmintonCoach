"""事件检测 v2 — 滑窗聚合 + 活动状态机 + 挥拍检测 + 多条件联合判定

架构：
  原始帧数据 → 滑窗聚合 → 活动状态机 → 挥拍检测 → 问题检测 → 事件合并输出
"""
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import EventConfig, SlowmoConfig
from ..models.enums import EventType, Severity
from ..models.schemas import FrameData, TimelineEvent


# ============================================================
# 1. 活动状态机
# ============================================================
class ActivityState(str, Enum):
    PLAYING = "playing"      # 正在击球/对抗
    MOVING = "moving"        # 走位/移动
    RESTING = "resting"      # 休息/暂停


@dataclass
class ActivityMetrics:
    """一个窗口内的聚合指标"""
    frame_start: int
    frame_end: int
    timestamp_start: float
    timestamp_end: float

    # 速度统计
    wrist_speed_mean: float = 0.0
    wrist_speed_peak: float = 0.0
    wrist_speed_std: float = 0.0

    # 加速度（速度变化率）
    wrist_accel_peak: float = 0.0

    # 身体倾斜
    body_lean_mean: float = 0.0
    body_lean_peak: float = 0.0

    # 关节角度（窗口内最极端值）
    joint_angles_min: Dict[str, float] = field(default_factory=dict)
    joint_angles_max: Dict[str, float] = field(default_factory=dict)

    # 活动状态
    state: ActivityState = ActivityState.RESTING

    # 置信度
    confidence: float = 0.0


# ============================================================
# 2. 滑窗聚合器
# ============================================================
class SlidingWindowAggregator:
    """将逐帧数据聚合为窗口级指标"""

    def __init__(self, window_size: int = 15, stride: int = 5):
        """
        Args:
            window_size: 窗口大小（帧数），15帧=0.5秒@30fps
            stride: 滑动步长（帧数），5帧=0.17秒@30fps
        """
        self.window_size = window_size
        self.stride = stride

    def aggregate(self, frames: List[FrameData]) -> List[ActivityMetrics]:
        """对全部帧做滑窗聚合"""
        if len(frames) < self.window_size:
            return []

        windows = []
        for i in range(0, len(frames) - self.window_size + 1, self.stride):
            window_frames = frames[i:i + self.window_size]
            metrics = self._compute_window_metrics(window_frames, i, i + self.window_size - 1)
            windows.append(metrics)

        return windows

    def _compute_window_metrics(
        self, frames: List[FrameData], start_idx: int, end_idx: int
    ) -> ActivityMetrics:
        """计算一个窗口内的聚合指标"""
        speeds = [f.wrist_speed for f in frames]
        leans = [f.body_lean for f in frames]

        # 加速度：速度的一阶差分
        accels = [abs(speeds[i] - speeds[i-1]) for i in range(1, len(speeds))]

        # 关节角度统计
        joint_keys = ["left_elbow", "right_elbow", "left_knee", "right_knee",
                       "left_shoulder", "right_shoulder"]
        angle_mins = {}
        angle_maxs = {}
        for key in joint_keys:
            vals = [getattr(f.joint_angles, key) for f in frames
                    if getattr(f.joint_angles, key) is not None]
            if vals:
                angle_mins[key] = min(vals)
                angle_maxs[key] = max(vals)

        return ActivityMetrics(
            frame_start=start_idx,
            frame_end=end_idx,
            timestamp_start=frames[0].timestamp,
            timestamp_end=frames[-1].timestamp,
            wrist_speed_mean=float(np.mean(speeds)),
            wrist_speed_peak=float(np.max(speeds)),
            wrist_speed_std=float(np.std(speeds)),
            wrist_accel_peak=float(np.max(accels)) if accels else 0.0,
            body_lean_mean=float(np.mean(leans)),
            body_lean_peak=float(np.max(leans)),
            joint_angles_min=angle_mins,
            joint_angles_max=angle_maxs,
        )


# ============================================================
# 3. 活动状态分类器
# ============================================================
class ActivityClassifier:
    """基于窗口指标分类活动状态"""

    def __init__(self, speed_threshold_low=50, speed_threshold_high=300,
                 accel_threshold=100):
        self.speed_low = speed_threshold_low
        self.speed_high = speed_threshold_high
        self.accel_thresh = accel_threshold

    def classify(self, window: ActivityMetrics) -> ActivityState:
        """分类单个窗口的活动状态"""
        speed = window.wrist_speed_mean
        peak = window.wrist_speed_peak
        accel = window.wrist_accel_peak

        # 高速+高加速度 → PLAYING
        if peak > self.speed_high and accel > self.accel_thresh:
            return ActivityState.PLAYING

        # 中等速度 → MOVING
        if speed > self.speed_low:
            return ActivityState.MOVING

        # 低速 → RESTING
        return ActivityState.RESTING

    def classify_all(self, windows: List[ActivityMetrics]) -> List[ActivityMetrics]:
        """分类所有窗口，并做时序平滑（防止状态抖动）"""
        for w in windows:
            w.state = self.classify(w)

        # 时序平滑：如果前后都是PLAYING，中间短暂的非PLAYING也改为PLAYING
        smoothed = self._smooth_states(windows)
        for w, s in zip(windows, smoothed):
            w.state = s

        return windows

    def _smooth_states(self, windows: List[ActivityMetrics],
                       min_duration: int = 3) -> List[ActivityState]:
        """状态平滑：短于min_duration个窗口的状态转换会被消除"""
        states = [w.state for w in windows]
        if len(states) < min_duration * 2:
            return states

        smoothed = states.copy()
        i = 0
        while i < len(smoothed):
            # 找到当前状态段的结束
            j = i + 1
            while j < len(smoothed) and smoothed[j] == smoothed[i]:
                j += 1

            # 如果段太短，用前后状态填充
            segment_len = j - i
            if segment_len < min_duration:
                prev_state = smoothed[i-1] if i > 0 else smoothed[i]
                next_state = smoothed[j] if j < len(smoothed) else smoothed[i]
                # 如果前后状态相同，用那个状态填充
                if prev_state == next_state:
                    for k in range(i, j):
                        smoothed[k] = prev_state
            i = j

        return smoothed


# ============================================================
# 4. 挥拍检测器
# ============================================================
@dataclass
class SwingEvent:
    """一次挥拍"""
    peak_frame: int
    peak_timestamp: float
    peak_speed: float
    peak_accel: float
    swing_type: str = "unknown"  # smash, clear, drop, net, etc.


class SwingDetector:
    """基于手腕速度峰值+加速度突变检测挥拍"""

    def __init__(self, speed_threshold=800, accel_threshold=300,
                 min_interval_frames=10):
        """
        Args:
            speed_threshold: 手腕速度峰值阈值
            accel_threshold: 加速度突变阈值
            min_interval_frames: 两次挥拍最小间隔（帧）
        """
        self.speed_thresh = speed_threshold
        self.accel_thresh = accel_threshold
        self.min_interval = min_interval_frames

    def detect(self, windows: List[ActivityMetrics]) -> List[SwingEvent]:
        """从窗口中检测挥拍事件"""
        swings = []

        for w in windows:
            if w.state != ActivityState.PLAYING:
                continue

            # 条件1: 手腕速度峰值超过阈值
            if w.wrist_speed_peak < self.speed_thresh:
                continue

            # 条件2: 加速度突变超过阈值（挥拍起始的标志）
            if w.wrist_accel_peak < self.accel_thresh:
                continue

            # 条件3: 与上次挥拍间隔足够
            peak_frame = w.frame_start + int(
                (w.frame_end - w.frame_start) * 0.5
            )
            if swings and (peak_frame - swings[-1].peak_frame) < self.min_interval:
                continue

            # 判断挥拍类型（基于速度特征）
            swing_type = self._classify_swing(w)

            swings.append(SwingEvent(
                peak_frame=peak_frame,
                peak_timestamp=w.timestamp_start + (w.timestamp_end - w.timestamp_start) * 0.5,
                peak_speed=w.wrist_speed_peak,
                peak_accel=w.wrist_accel_peak,
                swing_type=swing_type,
            ))

        return swings

    def _classify_swing(self, w: ActivityMetrics) -> str:
        """基于速度和角度特征粗略分类挥拍类型"""
        speed = w.wrist_speed_peak
        lean = w.body_lean_peak

        if speed > 1500 and lean > 40:
            return "smash"       # 杀球：高速+身体前倾
        elif speed > 1200:
            return "clear"       # 高远球：高速
        elif lean > 50:
            return "net_shot"    # 网前球：身体前倾大
        else:
            return "swing"       # 一般挥拍


# ============================================================
# 5. 问题检测器（仅在PLAYING状态下检测）
# ============================================================
@dataclass
class ProblemEvent:
    """一个技术问题"""
    start_frame: int
    end_frame: int
    timestamp: float
    joint: str
    value: float
    normal_range: Tuple[float, float]
    severity: str
    description: str
    improvement: str


class ProblemDetector:
    """在PLAYING状态下检测技术问题，要求最小持续时间"""

    def __init__(self, angle_ranges: Dict[str, List[float]],
                 min_duration_frames: int = 6,
                 tolerance: float = 10):
        """
        Args:
            angle_ranges: 关节角度正常范围
            min_duration_frames: 问题必须持续的最小帧数
            tolerance: 容差（度），超出范围tolerance°才标记
        """
        self.angle_ranges = {k: (v[0], v[1]) for k, v in angle_ranges.items()}
        self.min_duration = min_duration_frames
        self.tolerance = tolerance

    def detect(self, windows: List[ActivityMetrics]) -> List[ProblemEvent]:
        """在PLAYING窗口中检测持续性问题"""
        problems = []

        # 按关节收集异常窗口
        joint_violations: Dict[str, List[Tuple[ActivityMetrics, float, str]]] = {
            k: [] for k in self.angle_ranges
        }

        for w in windows:
            if w.state != ActivityState.PLAYING:
                continue

            for joint, (lo, hi) in self.angle_ranges.items():
                min_val = w.joint_angles_min.get(joint)
                max_val = w.joint_angles_max.get(joint)

                if min_val is not None and min_val < lo - self.tolerance:
                    joint_violations[joint].append((w, min_val, "too_low"))
                elif max_val is not None and max_val > hi + self.tolerance:
                    joint_violations[joint].append((w, max_val, "too_high"))

        # 合并连续的异常窗口为问题事件
        for joint, violations in joint_violations.items():
            if not violations:
                continue

            lo, hi = self.angle_ranges[joint]
            merged = self._merge_consecutive(violations)

            for group in merged:
                start_w, end_w, worst_val, direction = group
                duration_frames = end_w.frame_end - start_w.frame_start
                if duration_frames < self.min_duration:
                    continue

                if direction == "too_low":
                    desc = f"{joint}角度过小: {worst_val:.0f}°（正常{lo}-{hi}°）"
                    fix = f"注意{joint}不要过度弯曲"
                else:
                    desc = f"{joint}角度过大: {worst_val:.0f}°（正常{lo}-{hi}°）"
                    fix = f"注意{joint}不要过度伸展"

                severity = "critical" if abs(worst_val - (lo if direction == "too_low" else hi)) > 30 else "warning"

                problems.append(ProblemEvent(
                    start_frame=start_w.frame_start,
                    end_frame=end_w.frame_end,
                    timestamp=start_w.timestamp_start,
                    joint=joint,
                    value=worst_val,
                    normal_range=(lo, hi),
                    severity=severity,
                    description=desc,
                    improvement=fix,
                ))

        return problems

    def _merge_consecutive(self, violations):
        """合并连续的异常窗口"""
        if not violations:
            return []

        groups = []
        current = [violations[0]]

        for i in range(1, len(violations)):
            prev_w = current[-1][0]
            curr_w = violations[i][0]
            prev_dir = current[-1][2]
            curr_dir = violations[i][2]

            # 连续且同方向
            if (curr_w.frame_start <= prev_w.frame_end + 10 and
                    curr_dir == prev_dir):
                current.append(violations[i])
            else:
                groups.append(self._summarize_group(current))
                current = [violations[i]]

        groups.append(self._summarize_group(current))
        return groups

    def _summarize_group(self, group):
        """汇总一组连续异常"""
        start_w = group[0][0]
        end_w = group[-1][0]
        direction = group[0][2]
        if direction == "too_low":
            worst = min(v[1] for v in group)
        else:
            worst = max(v[1] for v in group)
        return (start_w, end_w, worst, direction)


# ============================================================
# 6. 主检测器（组合所有组件）
# ============================================================
class EventDetector:
    """事件检测器 v2 — 完整pipeline"""

    def __init__(self, event_config: EventConfig, slowmo_config: SlowmoConfig):
        self.slowmo_config = slowmo_config

        # 滑窗聚合器
        self.aggregator = SlidingWindowAggregator(window_size=15, stride=5)

        # 活动状态分类器
        self.activity_classifier = ActivityClassifier(
            speed_threshold_low=30,      # 降低：训练视频速度低
            speed_threshold_high=150,    # 降低：从300→150
            accel_threshold=50,          # 降低：从100→50
        )

        # 挥拍检测器
        self.swing_detector = SwingDetector(
            speed_threshold=300,         # 降低：从800→300（训练视频速度低）
            accel_threshold=100,         # 降低：从300→100
            min_interval_frames=15,
        )

        # 问题检测器
        self.problem_detector = ProblemDetector(
            angle_ranges=event_config.angle_ranges,
            min_duration_frames=6,
            tolerance=10,
        )

    def detect_all(self, frames: List[FrameData]) -> List[TimelineEvent]:
        """完整检测pipeline"""
        if len(frames) < 15:
            return []

        # Step 1: 滑窗聚合
        windows = self.aggregator.aggregate(frames)

        # Step 2: 活动状态分类
        windows = self.activity_classifier.classify_all(windows)

        # Step 3: 挥拍检测（精彩瞬间）
        swings = self.swing_detector.detect(windows)

        # Step 4: 问题检测（仅PLAYING状态）
        problems = self.problem_detector.detect(windows)

        # Step 5: 转换为TimelineEvent
        events = []

        for swing in swings:
            events.append(TimelineEvent(
                frame_idx=swing.peak_frame,
                timestamp=swing.peak_timestamp,
                event_type=EventType.HIGHLIGHT,
                sub_type=f"swing_{swing.swing_type}",
                severity=Severity.INFO,
                description=f"{swing.swing_type} 速度{swing.peak_speed:.0f}",
                improvement="",
                start_frame=max(0, swing.peak_frame - self.slowmo_config.pre_event_frames),
                end_frame=swing.peak_frame + self.slowmo_config.post_event_frames,
                score=min(100, swing.peak_speed / 15),
            ))

        for prob in problems:
            events.append(TimelineEvent(
                frame_idx=prob.start_frame,
                timestamp=prob.timestamp,
                event_type=EventType.PROBLEM,
                sub_type=f"{prob.joint}_{'low' if '过小' in prob.description else 'high'}",
                severity=Severity(prob.severity),
                description=prob.description,
                improvement=prob.improvement,
                start_frame=max(0, prob.start_frame - self.slowmo_config.pre_event_frames),
                end_frame=prob.end_frame + self.slowmo_config.post_event_frames,
                score=abs(prob.value - sum(prob.normal_range)/2) / (prob.normal_range[1] - prob.normal_range[0]) * 100,
            ))

        # 按时间排序
        events.sort(key=lambda e: e.timestamp)

        return events
