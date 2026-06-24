"""事件检测：问题 + 精彩瞬间"""
from typing import List

from ..config import EventConfig, SlowmoConfig
from ..models.enums import EventType, Severity
from ..models.schemas import FrameData, TimelineEvent


class EventDetector:
    def __init__(self, event_config: EventConfig, slowmo_config: SlowmoConfig):
        self.angle_ranges = event_config.angle_ranges
        self.thresholds = event_config.highlight_thresholds
        self.pre_frames = slowmo_config.pre_event_frames
        self.post_frames = slowmo_config.post_event_frames

    def detect_all(self, frames: List[FrameData]) -> List[TimelineEvent]:
        events: List[TimelineEvent] = []
        for fd in frames:
            events.extend(self._detect_frame(fd))
        return self._merge_events(events)

    def _detect_frame(self, fd: FrameData) -> List[TimelineEvent]:
        events: List[TimelineEvent] = []
        angles = fd.joint_angles.model_dump()

        # 问题检测：关节角度超范围
        for joint, (lo, hi) in self.angle_ranges.items():
            val = angles.get(joint)
            if val is None:
                continue
            if val < lo:
                events.append(
                    TimelineEvent(
                        frame_idx=fd.frame_idx,
                        timestamp=fd.timestamp,
                        event_type=EventType.PROBLEM,
                        sub_type=f"{joint}_too_closed",
                        severity=Severity.WARNING if val > lo * 0.8 else Severity.CRITICAL,
                        description=f"{joint}角度过小: {val:.0f}°（正常{lo}-{hi}°）",
                        improvement=f"注意{joint}不要过度弯曲，保持{lo}°以上",
                        start_frame=max(0, fd.frame_idx - self.pre_frames),
                        end_frame=fd.frame_idx + self.post_frames,
                        score=abs(val - lo) / lo * 100,
                    )
                )
            elif val > hi:
                events.append(
                    TimelineEvent(
                        frame_idx=fd.frame_idx,
                        timestamp=fd.timestamp,
                        event_type=EventType.PROBLEM,
                        sub_type=f"{joint}_overextend",
                        severity=Severity.WARNING if val < hi * 1.1 else Severity.CRITICAL,
                        description=f"{joint}角度过大: {val:.0f}°（正常{lo}-{hi}°）",
                        improvement=f"注意{joint}不要过度伸展，保持{hi}°以内",
                        start_frame=max(0, fd.frame_idx - self.pre_frames),
                        end_frame=fd.frame_idx + self.post_frames,
                        score=abs(val - hi) / hi * 100,
                    )
                )

        # 精彩瞬间：高速挥拍
        if fd.wrist_speed > self.thresholds["wrist_speed_max"]:
            events.append(
                TimelineEvent(
                    frame_idx=fd.frame_idx,
                    timestamp=fd.timestamp,
                    event_type=EventType.HIGHLIGHT,
                    sub_type="fast_swing",
                    severity=Severity.INFO,
                    description=f"高速挥拍！手腕速度 {fd.wrist_speed:.0f}",
                    improvement="",
                    start_frame=max(0, fd.frame_idx - self.pre_frames),
                    end_frame=fd.frame_idx + self.post_frames,
                    score=min(100, fd.wrist_speed / 20),
                )
            )

        # 精彩瞬间：极限救球
        if fd.body_lean > self.thresholds["body_lean_max"]:
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

        return events

    def _merge_events(
        self, events: List[TimelineEvent], gap: int = 60
    ) -> List[TimelineEvent]:
        if not events:
            return []
        events.sort(key=lambda e: e.start_frame)
        merged = [events[0]]
        for ev in events[1:]:
            last = merged[-1]
            if (
                ev.event_type == last.event_type
                and ev.start_frame <= last.end_frame + gap
            ):
                last.end_frame = max(last.end_frame, ev.end_frame)
                last.score = max(last.score, ev.score)
            else:
                merged.append(ev)
        return merged
