"""慢镜头生成（精简版：纯慢放+时间戳）"""
from typing import List

import cv2
import numpy as np

from ..config import SlowmoConfig
from ..models.schemas import TimelineEvent


def generate_slowmo_clips(
    video_path: str,
    events: List[TimelineEvent],
    output_dir: str,
    config: SlowmoConfig,
) -> tuple[str, str]:
    """生成问题集锦和精彩集锦"""
    problem_events = [e for e in events if e.event_type.value == "problem"]
    highlight_events = [e for e in events if e.event_type.value == "highlight"]

    problem_path = f"{output_dir}/problems_slowmo.mp4"
    highlight_path = f"{output_dir}/highlights_slowmo.mp4"

    _concat_slowmo(video_path, problem_events, problem_path, config)
    _concat_slowmo(video_path, highlight_events, highlight_path, config)

    return problem_path, highlight_path


def _concat_slowmo(
    video_path: str,
    events: List[TimelineEvent],
    output_path: str,
    config: SlowmoConfig,
):
    if not events:
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480)
        )
        writer.release()
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = fps * config.factor

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
    events.sort(key=lambda e: e.start_frame)

    for ev in events:
        cap.set(cv2.CAP_PROP_POS_FRAMES, ev.start_frame)
        for _i in range(ev.start_frame, ev.end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            # 只标注时间戳（右上角，小字）
            ts = f"{ev.timestamp:.1f}s"
            cv2.putText(frame, ts, (w - 80, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)

    writer.release()
    cap.release()
