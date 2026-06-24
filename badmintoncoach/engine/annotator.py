"""视频标注：骨骼 + 球 + 指标HUD（支持中文）"""
from typing import List

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
COLOR_TEXT = (255, 255, 255)   # 白色文字
COLOR_BG = (0, 0, 0)           # 黑色背景
COLOR_WARNING = (0, 100, 255)  # 橙色警告
COLOR_CRITICAL = (0, 0, 255)   # 红色严重

# PIL字体缓存
_font_cache = {}


def _get_font(size=18):
    """获取中文PIL字体"""
    if size not in _font_cache:
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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
    """用PIL渲染中文文字到OpenCV图像"""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(size)
    draw.text(pos, text, fill=color, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def annotate_frame(
    frame: np.ndarray, frame_data: FrameData, kpt_threshold: float = 0.43
) -> np.ndarray:
    """在单帧上绘制标注"""
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
            # 标注关键点序号（调试用）
            cv2.putText(img, str(i), (pt[0]+5, pt[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

    # 绘制球
    if frame_data.ball_position is not None:
        bp = tuple(np.array(frame_data.ball_position).astype(int))
        cv2.circle(img, bp, 8, COLOR_BALL, -1, cv2.LINE_AA)
        cv2.circle(img, bp, 12, COLOR_BALL, 2, cv2.LINE_AA)

    # HUD 指标（PIL中文渲染）
    img = draw_hud(img, frame_data)
    return img


def draw_hud(img: np.ndarray, fd: FrameData) -> np.ndarray:
    """绘制指标HUD（中文）"""
    h, w = img.shape[:2]

    # 半透明背景条
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (300, 220), COLOR_BG, -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    # 用PIL渲染中文
    lines = [
        f"帧: {fd.frame_idx}  时间: {fd.timestamp:.1f}s",
    ]

    angles = fd.joint_angles
    if angles.right_elbow is not None:
        lines.append(f"右肘: {angles.right_elbow:.0f}°")
    if angles.left_elbow is not None:
        lines.append(f"左肘: {angles.left_elbow:.0f}°")
    if angles.right_knee is not None:
        lines.append(f"右膝: {angles.right_knee:.0f}°")
    if angles.left_knee is not None:
        lines.append(f"左膝: {angles.left_knee:.0f}°")
    if angles.right_shoulder is not None:
        lines.append(f"右肩: {angles.right_shoulder:.0f}°")
    if angles.left_shoulder is not None:
        lines.append(f"左肩: {angles.left_shoulder:.0f}°")

    lines.append(f"手腕速度: {fd.wrist_speed:.0f} px/f")
    lines.append(f"身体倾斜: {fd.body_lean:.0f}°")

    y = 8
    for line in lines:
        img = _put_text_cn(img, line, (10, y), COLOR_TEXT, 16)
        y += 24

    return img


def annotate_video(
    input_path: str,
    output_path: str,
    frames_data: List[FrameData],
    progress_callback=None,
) -> str:
    """标注完整视频"""
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
        annotated = annotate_frame(frame, fd)
        writer.write(annotated)
        if progress_callback and i % 30 == 0:
            progress_callback(i / len(frames_data))

    writer.release()
    cap.release()
    return output_path
