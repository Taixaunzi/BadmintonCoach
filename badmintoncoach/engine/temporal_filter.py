"""第一层时序过滤：滑窗聚合 + 活动状态机 + 挥拍检测 + 多条件联合

解决的核心问题：
- 逐帧检测噪声太多 → 滑窗聚合看统计量
- 走位/休息也触发事件 → 活动状态机区分打球 vs 非打球
- 单帧速度抖动 → 挥拍检测用加速度峰值
- 条件太松 → 多条件联合触发
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from ..models.schemas import FrameData


# ============================================================
# 1. 滑窗聚合器
# ============================================================
@dataclass
class WindowStats:
    """一个滑动窗口内的统计量"""
    mean: float = 0.0
    peak: float = 0.0
    valley: float = float("inf")
    std: float = 0.0
    range: float = 0.0
    count: int = 0


class SlidingWindow:
    """定长滑动窗口，维护 N 帧的统计量"""

    def __init__(self, size: int = 15):
        self.size = size
        self._buf: deque = deque(maxlen=size)

    def push(self, value: float) -> None:
        self._buf.append(value)

    def stats(self) -> WindowStats:
        if not self._buf:
            return WindowStats()
        arr = np.array(self._buf, dtype=float)
        return WindowStats(
            mean=float(np.mean(arr)),
            peak=float(np.max(arr)),
            valley=float(np.min(arr)),
            std=float(np.std(arr)),
            range=float(np.ptp(arr)),
            count=len(arr),
        )

    @property
    def full(self) -> bool:
        return len(self._buf) >= self.size

    def clear(self) -> None:
        self._buf.clear()


# ============================================================
# 2. 活动状态机
# ============================================================
class ActivityState(str, Enum):
    IDLE = "idle"           # 静止/休息
    MOVING = "moving"       # 走位（低速移动）
    PLAYING = "playing"     # 打球中（高速运动）
    COOLDOWN = "cooldown"   # 刚打完一拍，过渡期


class ActivityStateMachine:
    """基于运动强度区分打球 vs 走位 vs 休息

    判定逻辑：
    - wrist_speed 滑窗均值 > playing_threshold → PLAYING
    - wrist_speed 滑窗均值 > moving_threshold  → MOVING
    - 否则 → IDLE
    - PLAYING 后有 cooldown_frames 的 COOLDOWN 过渡
    """

    def __init__(
        self,
        playing_threshold: float = 800.0,
        moving_threshold: float = 200.0,
        cooldown_frames: int = 10,
    ):
        self.playing_threshold = playing_threshold
        self.moving_threshold = moving_threshold
        self.cooldown_frames = cooldown_frames
        self.state = ActivityState.IDLE
        self._cooldown_counter = 0

    def update(self, wrist_speed_mean: float) -> ActivityState:
        # COOLDOWN 倒计时
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            if self._cooldown_counter == 0:
                self.state = ActivityState.IDLE
            return self.state

        if wrist_speed_mean >= self.playing_threshold:
            self.state = ActivityState.PLAYING
        elif wrist_speed_mean >= self.moving_threshold:
            # 从 PLAYING 降级时先进入 COOLDOWN
            if self.state == ActivityState.PLAYING:
                self.state = ActivityState.COOLDOWN
                self._cooldown_counter = self.cooldown_frames
            else:
                self.state = ActivityState.MOVING
        else:
            if self.state == ActivityState.PLAYING:
                self.state = ActivityState.COOLDOWN
                self._cooldown_counter = self.cooldown_frames
            else:
                self.state = ActivityState.IDLE

        return self.state

    @property
    def is_active(self) -> bool:
        """是否处于有意义的运动状态（打球中 或 冷却期）"""
        return self.state in (ActivityState.PLAYING, ActivityState.COOLDOWN)


# ============================================================
# 3. 挥拍检测器
# ============================================================
@dataclass
class SwingEvent:
    """一次挥拍事件"""
    peak_frame: int
    peak_speed: float
    acceleration: float   # 速度变化率峰值
    wrist: str            # "left" / "right"


class SwingDetector:
    """手腕速度峰值 + 加速度突变 → 检测挥拍时刻

    检测逻辑：
    1. 速度从低点上升超过 speed_threshold
    2. 加速度（帧间速度差）超过 accel_threshold
    3. 达到局部峰值后标记为挥拍
    4. 有最小间隔（min_gap_frames）防止重复计数
    """

    def __init__(
        self,
        speed_threshold: float = 600.0,
        accel_threshold: float = 300.0,
        min_gap_frames: int = 8,
    ):
        self.speed_threshold = speed_threshold
        self.accel_threshold = accel_threshold
        self.min_gap_frames = min_gap_frames
        self._prev_speed: float = 0.0
        self._prev_accel: float = 0.0
        self._last_swing_frame: int = -999
        self._rising: bool = False
        self._peak_speed: float = 0.0
        self._peak_frame: int = 0

    def update(self, frame_idx: int, wrist_speed: float) -> Optional[SwingEvent]:
        accel = wrist_speed - self._prev_speed
        swing = None

        # 检测上升沿
        if accel > 0 and wrist_speed > self.speed_threshold:
            if not self._rising:
                self._rising = True
                self._peak_speed = wrist_speed
                self._peak_frame = frame_idx
            elif wrist_speed > self._peak_speed:
                self._peak_speed = wrist_speed
                self._peak_frame = frame_idx
        elif self._rising and accel <= 0:
            # 速度开始下降 → 峰值已过
            self._rising = False
            if (
                self._peak_speed >= self.speed_threshold
                and self._prev_accel >= self.accel_threshold
                and (self._peak_frame - self._last_swing_frame) >= self.min_gap_frames
            ):
                swing = SwingEvent(
                    peak_frame=self._peak_frame,
                    peak_speed=self._peak_speed,
                    acceleration=self._prev_accel,
                    wrist="right",  # 简化：后续可区分左右手
                )
                self._last_swing_frame = self._peak_frame

        self._prev_speed = wrist_speed
        self._prev_accel = accel
        return swing

    def reset(self) -> None:
        self._prev_speed = 0.0
        self._prev_accel = 0.0
        self._rising = False
        self._peak_speed = 0.0


# ============================================================
# 4. 角度异常检测器（带最小持续时间）
# ============================================================
@dataclass
class AngleAnomaly:
    """持续性的角度异常"""
    joint: str
    direction: str        # "too_closed" / "overextend"
    start_frame: int
    end_frame: int
    peak_value: float     # 最极端的值
    severity_score: float # 越大越严重


class AngleAnomalyDetector:
    """角度异常必须持续 min_duration 帧才算

    逻辑：
    - 每帧记录各关节是否超限
    - 连续超限 >= min_duration → 标记为异常区间
    - 区间内记录最极端值
    """

    def __init__(self, min_duration: int = 5):
        self.min_duration = min_duration
        # {joint: {"direction": str, "start": int, "extreme": float}}
        self._active: Dict[str, dict] = {}

    def update(
        self,
        frame_idx: int,
        joint: str,
        value: float,
        lo: float,
        hi: float,
    ) -> Optional[AngleAnomaly]:
        anomaly = None

        if value < lo:
            direction = "too_closed"
            severity = (lo - value) / lo
        elif value > hi:
            direction = "overextend"
            severity = (value - hi) / hi
        else:
            # 角度正常 → 结束当前异常区间
            if joint in self._active:
                a = self._active.pop(joint)
                duration = frame_idx - a["start"]
                if duration >= self.min_duration:
                    anomaly = AngleAnomaly(
                        joint=joint,
                        direction=a["direction"],
                        start_frame=a["start"],
                        end_frame=frame_idx,
                        peak_value=a["extreme"],
                        severity_score=a["severity"],
                    )
            return anomaly

        # 还在异常中
        if joint not in self._active:
            self._active[joint] = {
                "direction": direction,
                "start": frame_idx,
                "extreme": value,
                "severity": severity,
            }
        else:
            entry = self._active[joint]
            if direction != entry["direction"]:
                # 方向切换 → 结束旧的，开始新的
                duration = frame_idx - entry["start"]
                if duration >= self.min_duration:
                    anomaly = AngleAnomaly(
                        joint=joint,
                        direction=entry["direction"],
                        start_frame=entry["start"],
                        end_frame=frame_idx,
                        peak_value=entry["extreme"],
                        severity_score=entry["severity"],
                    )
                self._active[joint] = {
                    "direction": direction,
                    "start": frame_idx,
                    "extreme": value,
                    "severity": severity,
                }
            else:
                # 更新极值
                if direction == "too_closed":
                    entry["extreme"] = min(entry["extreme"], value)
                else:
                    entry["extreme"] = max(entry["extreme"], value)
                entry["severity"] = max(entry["severity"], severity)

        return anomaly

    def flush(self, frame_idx: int) -> List[AngleAnomaly]:
        """流式处理结束时，把还在进行中的异常都输出"""
        results = []
        for joint, a in self._active.items():
            duration = frame_idx - a["start"]
            if duration >= self.min_duration:
                results.append(
                    AngleAnomaly(
                        joint=joint,
                        direction=a["direction"],
                        start_frame=a["start"],
                        end_frame=frame_idx,
                        peak_value=a["extreme"],
                        severity_score=a["severity"],
                    )
                )
        self._active.clear()
        return results


# ============================================================
# 5. 组合：TemporalFilter（对外接口）
# ============================================================
@dataclass
class TemporalConfig:
    """时序过滤配置"""
    # 滑窗
    window_size: int = 15
    # 状态机
    playing_speed_threshold: float = 800.0
    moving_speed_threshold: float = 200.0
    cooldown_frames: int = 10
    # 挥拍
    swing_speed_threshold: float = 600.0
    swing_accel_threshold: float = 300.0
    swing_min_gap: int = 8
    # 角度
    angle_min_duration: int = 5
    # 多条件联合
    highlight_requires_playing: bool = True  # 精彩瞬间只在打球状态触发


class TemporalFilter:
    """第一层时序过滤的统一入口

    用法：
        tf = TemporalFilter(config)
        for frame in frames:
            result = tf.process_frame(frame)
            # result 包含：activity_state, swing_events, angle_anomalies
        result = tf.flush()  # 处理剩余
    """

    def __init__(self, config: TemporalConfig):
        self.config = config
        self.speed_window = SlidingWindow(config.window_size)
        self.activity = ActivityStateMachine(
            playing_threshold=config.playing_speed_threshold,
            moving_threshold=config.moving_speed_threshold,
            cooldown_frames=config.cooldown_frames,
        )
        self.swing_detector = SwingDetector(
            speed_threshold=config.swing_speed_threshold,
            accel_threshold=config.swing_accel_threshold,
            min_gap_frames=config.swing_min_gap,
        )
        self.angle_detectors: Dict[str, AngleAnomalyDetector] = {}
        self._frame_count = 0

    def process_frame(self, fd: FrameData, angle_ranges: Dict[str, List[float]]):
        """处理一帧，返回时序过滤结果"""
        self._frame_count = fd.frame_idx

        # 1. 滑窗聚合
        self.speed_window.push(fd.wrist_speed)
        speed_stats = self.speed_window.stats()

        # 2. 活动状态机
        activity_state = self.activity.update(speed_stats.mean)

        # 3. 挥拍检测
        swing = self.swing_detector.update(fd.frame_idx, fd.wrist_speed)

        # 4. 角度异常检测（带最小持续时间）
        angle_anomalies = []
        angles = fd.joint_angles.model_dump()
        for joint, (lo, hi) in angle_ranges.items():
            val = angles.get(joint)
            if val is None:
                continue
            if joint not in self.angle_detectors:
                self.angle_detectors[joint] = AngleAnomalyDetector(
                    min_duration=self.config.angle_min_duration
                )
            anomaly = self.angle_detectors[joint].update(fd.frame_idx, joint, val, lo, hi)
            if anomaly:
                angle_anomalies.append(anomaly)

        return FrameFilterResult(
            frame_idx=fd.frame_idx,
            timestamp=fd.timestamp,
            activity_state=activity_state,
            speed_stats=speed_stats,
            swing=swing,
            angle_anomalies=angle_anomalies,
        )

    def flush(self) -> List[AngleAnomaly]:
        """流结束时刷新所有进行中的角度异常"""
        all_anomalies = []
        for detector in self.angle_detectors.values():
            all_anomalies.extend(detector.flush(self._frame_count))
        return all_anomalies


@dataclass
class FrameFilterResult:
    """单帧的时序过滤结果"""
    frame_idx: int
    timestamp: float
    activity_state: ActivityState
    speed_stats: WindowStats
    swing: Optional[SwingEvent] = None
    angle_anomalies: List[AngleAnomaly] = field(default_factory=list)
