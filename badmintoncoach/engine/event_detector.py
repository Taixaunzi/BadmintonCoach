"""事件检测（v2）：接入第一层时序过滤

改进：
- 不再逐帧检测 → 通过 TemporalFilter 滑窗+状态机过滤
- 问题检测：角度异常必须持续 N 帧
- 精彩瞬间：挥拍检测（速度峰值+加速度）替代单帧阈值
- 多条件联合：精彩瞬间只在"打球状态"触发
"""

from typing import Dict, List

from ..config import EventConfig, SlowmoConfig
from ..models.enums import EventType, Severity
from ..models.schemas import FrameData, TimelineEvent
from .temporal_filter import (
    ActivityState,
    AngleAnomaly,
    SwingEvent,
    TemporalConfig,
    TemporalFilter,
)


class EventDetector:
    def __init__(self, event_config: EventConfig, slowmo_config: SlowmoConfig):
        self.angle_ranges = event_config.angle_ranges
        self.thresholds = event_config.highlight_thresholds
        self.pre_frames = slowmo_config.pre_event_frames
        self.post_frames = slowmo_config.post_event_frames

        # 初始化时序过滤器
        temporal_cfg = self._build_temporal_config(event_config)
        self.temporal = TemporalFilter(temporal_cfg)

    def _build_temporal_config(self, ec: EventConfig) -> TemporalConfig:
        """从 EventConfig 推导时序过滤参数"""
        thresh = ec.highlight_thresholds
        return TemporalConfig(
            window_size=15,
            playing_speed_threshold=thresh.get("wrist_speed_max", 1500) * 0.5,
            moving_speed_threshold=thresh.get("wrist_speed_max", 1500) * 0.15,
            cooldown_frames=10,
            swing_speed_threshold=thresh.get("wrist_speed_max", 1500) * 0.4,
            swing_accel_threshold=thresh.get("speed_change_rate", 500) * 0.6,
            swing_min_gap=8,
            angle_min_duration=5,
            highlight_requires_playing=True,
        )

    def detect_all(self, frames: List[FrameData]) -> List[TimelineEvent]:
        events: List[TimelineEvent] = []

        for fd in frames:
            result = self.temporal.process_frame(fd, self.angle_ranges)

            # 1. 挥拍事件 → 精彩瞬间
            if result.swing:
                ev = self._swing_to_event(result.swing, result.activity_state)
                if ev:
                    events.append(ev)

            # 2. 持续性角度异常 → 问题
            for anomaly in result.angle_anomalies:
                ev = self._anomaly_to_event(anomaly)
                if ev:
                    events.append(ev)

            # 3. 极限救球（保留，但也加状态机过滤）
            if (
                result.activity_state == ActivityState.PLAYING
                and fd.body_lean > self.thresholds["body_lean_max"]
            ):
                events.append(
                    TimelineEvent(
                        frame_idx=fd.frame_idx,
                        timestamp=fd.timestamp,
                        event_type=EventType.HIGHLIGHT,
                        sub_type="extreme_lean",
                        severity=Severity.INFO,
                        description=f"极限救球！身体倾斜 {fd.body_lean:.0f}°",
                        improvement="",
                        start_frame=max(0, fd.frame_idx - self.pre_frames),
                        end_frame=fd.frame_idx + self.post_frames,
                        score=min(100, fd.body_lean * 2),
                    )
                )

        # 流结束：刷出剩余角度异常
        remaining = self.temporal.flush()
        for anomaly in remaining:
            ev = self._anomaly_to_event(anomaly)
            if ev:
                events.append(ev)

        return self._merge_events(events)

    def _swing_to_event(self, swing: SwingEvent, state: ActivityState) -> TimelineEvent | None:
        """挥拍 → 精彩瞬间事件"""
        # 多条件联合：只在打球状态触发
        if self.temporal.config.highlight_requires_playing and state != ActivityState.PLAYING:
            return None

        # 评分：速度越高、加速度越大 → 分越高
        speed_score = min(100, swing.peak_speed / 20)
        accel_score = min(100, swing.acceleration / 10)
        combined_score = speed_score * 0.6 + accel_score * 0.4

        return TimelineEvent(
            frame_idx=swing.peak_frame,
            timestamp=swing.peak_frame / 30.0,  # 近似，后续由调用方修正
            event_type=EventType.HIGHLIGHT,
            sub_type="fast_swing",
            severity=Severity.INFO,
            description=(
                f"高速挥拍！速度 {swing.peak_speed:.0f}，"
                f"加速度 {swing.acceleration:.0f}"
            ),
            improvement="",
            start_frame=max(0, swing.peak_frame - self.pre_frames),
            end_frame=swing.peak_frame + self.post_frames,
            score=combined_score,
        )

    def _anomaly_to_event(self, anomaly: AngleAnomaly) -> TimelineEvent:
        """角度异常 → 问题事件"""
        joint = anomaly.joint
        lo, hi = self.angle_ranges.get(joint, [0, 180])

        if anomaly.direction == "too_closed":
            desc = f"{joint}角度持续过小: {anomaly.peak_value:.0f}°（正常{lo}-{hi}°）"
            imp = f"注意{joint}不要过度弯曲，保持{lo}°以上"
            severity = Severity.CRITICAL if anomaly.severity_score > 0.2 else Severity.WARNING
        else:
            desc = f"{joint}角度持续过大: {anomaly.peak_value:.0f}°（正常{lo}-{hi}°）"
            imp = f"注意{joint}不要过度伸展，保持{hi}°以内"
            severity = Severity.CRITICAL if anomaly.severity_score > 0.15 else Severity.WARNING

        # 时长越长、偏差越大 → 分越高
        duration = anomaly.end_frame - anomaly.start_frame
        score = min(100, anomaly.severity_score * 200 + duration * 2)

        return TimelineEvent(
            frame_idx=(anomaly.start_frame + anomaly.end_frame) // 2,
            timestamp=((anomaly.start_frame + anomaly.end_frame) / 2) / 30.0,
            event_type=EventType.PROBLEM,
            sub_type=f"{joint}_{anomaly.direction}",
            severity=severity,
            description=desc,
            improvement=imp,
            start_frame=anomaly.start_frame,
            end_frame=anomaly.end_frame,
            score=score,
        )

    def _merge_events(
        self, events: List[TimelineEvent], gap: int = 30
    ) -> List[TimelineEvent]:
        """合并相邻同类型事件（v2: 更短的gap，因为时序过滤已减少噪声）"""
        if not events:
            return []
        events.sort(key=lambda e: e.start_frame)
        merged = [events[0]]
        for ev in events[1:]:
            last = merged[-1]
            if (
                ev.event_type == last.event_type
                and ev.sub_type == last.sub_type
                and ev.start_frame <= last.end_frame + gap
            ):
                last.end_frame = max(last.end_frame, ev.end_frame)
                last.score = max(last.score, ev.score)
            else:
                merged.append(ev)
        return merged
