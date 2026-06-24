"""视频标注：骨骼 + 球轨迹 + 球速度 + 指标HUD（中文）"""
from collections import deque
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..models.schemas import FrameData

# COCO骨架连接
SKELETON = [
    (5, 7), (7, 9),      # 左肩-左肘-左腕
    (6, 8), (8, 10),     # 右肩-右肘-右腕
    (5, 6),              # 左肩-右肩
    (5, 11), (6, 12),    # 肩-髋
    (11, 13), (13, 15),  # 左髋-左膝-左踝
    (12, 14), (14, 16),  # 右髋-右膝-右踝
    (11, 12),            # 左髋-右髋
]

COLOR_BONE = (0, 255, 0)       # 绿色骨架
COLOR_KPT = (0, 0, 255)        # 红色关键点
COLOR_BALL = (0, 165, 255)     # 橙色球
COLOR_TRAIL = (0, 200, 255)    # 黄色轨迹
COLOR_TEXT = (255, 255, 255)   # 白色文字
COLOR_BG = (0, 0, 0)           # 黑色背景

_font_cache = {}
_trail_history: deque = deque(maxlen=20)


def _get_font(size=18):
    if size not in _font_cache:
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]
        for fp in font_paths:
            try:
                _font_cache[size] = ImageFont.truetype(fp, size)
                break
            except (OSError, IOError):
                continue
        if size not in _font_cache:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def _put_text_cn(img: np.ndarray, text: str, pos: tuple,
                 color=(255, 255, 255), size=18) -> np.ndarray:
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(size)
    draw.text(pos, text, fill=color, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def annotate_frame(
    frame: np.ndarray,
    frame_data: FrameData,
    kpt_threshold: float = 0.43,
    ball_pos: Optional[np.ndarray] = None,
    ball_speed_kmh: float = 0.0,
) -> np.ndarray:
    """在单帧上绘制标注（骨骼+球轨迹+速度+HUD）"""
    img = frame.copy()
    kpts = np.array(frame_data.keypoints)
    scores = np.array(frame_data.scores)

    # 绘制骨架
    for i, j in SKELETON:
        if i < len(scores) and j < len(scores):
            if scores[i] > kpt_threshold and scores[j] > kpt_threshold:
                pt1 = tuple(kpts[i].astype(int))
                pt2 = tuple(kpts[j].astype(int))
                cv2.line(img, pt1, pt2, COLOR_BONE, 2, cv2.LINE_AA)

    # 绘制关键点
    for i in range(len(kpts)):
        if scores[i] > kpt_threshold:
            pt = tuple(kpts[i].astype(int))
            cv2.circle(img, pt, 4, COLOR_KPT, -1, cv2.LINE_AA)

    # 绘制球轨迹
    if ball_pos is not None:
        _trail_history.append(ball_pos.copy())
    _draw_trail(img)

    # 绘制当前球位置
    if ball_pos is not None:
        bp = tuple(ball_pos.astype(int))
        cv2.circle(img, bp, 6, COLOR_BALL, -1, cv2.LINE_AA)
        cv2.circle(img, bp, 10, COLOR_BALL, 2, cv2.LINE_AA)

        # 球速度标签
        if ball_speed_kmh > 5:
            speed_text = f"{ball_speed_kmh:.0f} km/h"
            cv2.putText(img, speed_text, (bp[0] + 15, bp[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BALL, 1, cv2.LINE_AA)

    # HUD指标
    img = draw_hud(img, frame_data, ball_speed_kmh)
    return img


def _draw_trail(img: np.ndarray):
    """绘制球轨迹（渐变色）"""
    if len(_trail_history) < 2:
        return
    for i in range(1, len(_trail_history)):
        p1 = _trail_history[i - 1]
        p2 = _trail_history[i]
        if p1 is None or p2 is None:
            continue
        # 越新的点越亮
        alpha = i / len(_trail_history)
        color = (
            int(50 + 200 * alpha),  # B
            int(100 + 155 * alpha), # G
            int(200 + 55 * alpha),  # R
        )
        pt1 = tuple(p1.astype(int))
        pt2 = tuple(p2.astype(int))
        cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)


def draw_hud(img: np.ndarray, fd: FrameData, ball_speed_kmh: float = 0.0) -> np.ndarray:
    """绘制指标HUD"""
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (300, 240), COLOR_BG, -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    lines = [f"帧: {fd.frame_idx}  时间: {fd.timestamp:.1f}s"]

    angles = fd.joint_angles
    if angles.right_elbow is not None:
        lines.append(f"右肘: {angles.right_elbow:.0f}°")
    if angles.left_elbow is not None:
        lines.append(f"左肘: {angles.left_elbow:.0f}°")
    if angles.right_knee is not None:
        lines.append(f"右膝: {angles.right_knee:.0f}°")
    if angles.left_knee is not None:
        lines.append(f"左膝: {angles.left_knee:.0f}°")
    lines.append(f"手腕速度: {fd.wrist_speed:.0f} px/f")
    lines.append(f"身体倾斜: {fd.body_lean:.0f}°")
    if ball_speed_kmh > 5:
        lines.append(f"球速: {ball_speed_kmh:.0f} km/h")

    y = 8
    for line in lines:
        img = _put_text_cn(img, line, (10, y), COLOR_TEXT, 16)
        y += 24

    return img


def reset_trail():
    """重置轨迹历史"""
    _trail_history.clear()


def annotate_video(
    input_path: str,
    output_path: str,
    frames_data: List[FrameData],
    ball_positions: Optional[List[Optional[np.ndarray]]] = None,
    ball_speeds: Optional[List[float]] = None,
    progress_callback=None,
) -> str:
    """标注完整视频（含球轨迹）"""
    reset_trail()
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i, fd in enumerate(frames_data):
        ret, frame = cap.read()
        if not ret:
            break

        bp = None
        bs = 0.0
        if ball_positions and i < len(ball_positions):
            bp = ball_positions[i]
        if ball_speeds and i < len(ball_speeds):
            bs = ball_speeds[i]

        annotated = annotate_frame(frame, fd, ball_pos=bp, ball_speed_kmh=bs)
        writer.write(annotated)
        if progress_callback and i % 30 == 0:
            progress_callback(i / len(frames_data))

    writer.release()
    cap.release()
    return output_path
