"""慢镜头生成"""
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
        # 无事件，创建空占位文件
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
        # 标题卡（1.5秒）
        title = np.zeros((h, w, 3), dtype=np.uint8)
        color = (0, 0, 255) if ev.event_type.value == "problem" else (0, 200, 255)
        label = "问题" if ev.event_type.value == "problem" else "精彩"
        text = f"{label}: {ev.description[:30]}"
        cv2.putText(
            title, text, (w // 6, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2
        )
        for _ in range(int(out_fps * 1.5)):
            writer.write(title)

        # 慢放片段
        cap.set(cv2.CAP_PROP_POS_FRAMES, ev.start_frame)
        for _i in range(ev.start_frame, ev.end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            # 叠加文字
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
            cv2.putText(
                frame,
                ev.description[:60],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            if ev.improvement:
                cv2.putText(
                    frame,
                    f"改进: {ev.improvement[:50]}",
                    (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                )
            writer.write(frame)

    writer.release()
    cap.release()
