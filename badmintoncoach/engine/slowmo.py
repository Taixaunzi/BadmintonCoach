"""慢镜头生成（支持中文）"""
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import SlowmoConfig
from ..models.schemas import TimelineEvent


def _get_font(size=24):
    """获取中文字体"""
    font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _put_text_cn(img: np.ndarray, text: str, pos: tuple,
                 color=(255, 255, 255), size=24) -> np.ndarray:
    """PIL渲染中文"""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(size)
    draw.text(pos, text, fill=color, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


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
        # 标题卡（1.5秒）
        title = np.zeros((h, w, 3), dtype=np.uint8)
        color = (0, 0, 255) if ev.event_type.value == "problem" else (0, 200, 255)
        label = "⚠ 问题" if ev.event_type.value == "problem" else "★ 精彩"
        text = f"{label}: {ev.description[:30]}"
        title = _put_text_cn(title, text, (w // 6, h // 2), color, 28)
        for _ in range(int(out_fps * 1.5)):
            writer.write(title)

        # 慢放片段
        cap.set(cv2.CAP_PROP_POS_FRAMES, ev.start_frame)
        for _i in range(ev.start_frame, ev.end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            # 半透明叠加条
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
            # 中文文字
            frame = _put_text_cn(frame, ev.description[:50], (10, 10), (255, 255, 255), 20)
            if ev.improvement:
                frame = _put_text_cn(frame, f"建议: {ev.improvement[:40]}", 
                                    (10, h - 35), (0, 255, 0), 18)
            writer.write(frame)

    writer.release()
    cap.release()
