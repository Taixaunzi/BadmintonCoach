"""视频标注：骨骼 + 球 + 指标HUD"""
from typing import List

import cv2
import numpy as np

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


def annotate_frame(
    frame: np.ndarray, frame_data: FrameData, kpt_threshold: float = 0.43
) -> np.ndarray:
    """在单帧上绘制标注"""
    img = frame.copy()
    kpts = np.array(frame_data.keypoints)
    scores = np.array(frame_data.scores)

    # 绘制骨架
    for i, j in SKELETON:
        if scores[i] > kpt_threshold and scores[j] > kpt_threshold:
            pt1 = tuple(kpts[i].astype(int))
            pt2 = tuple(kpts[j].astype(int))
            cv2.line(img, pt1, pt2, COLOR_BONE, 2, cv2.LINE_AA)

    # 绘制关键点
    for i in range(len(kpts)):
        if scores[i] > kpt_threshold:
            cv2.circle(img, tuple(kpts[i].astype(int)), 4, COLOR_KPT, -1, cv2.LINE_AA)

    # 绘制球
    if frame_data.ball_position is not None:
        bp = tuple(np.array(frame_data.ball_position).astype(int))
        cv2.circle(img, bp, 8, COLOR_BALL, -1, cv2.LINE_AA)
        cv2.circle(img, bp, 12, COLOR_BALL, 2, cv2.LINE_AA)

    # HUD 指标
    img = draw_hud(img, frame_data)
    return img


def draw_hud(img: np.ndarray, fd: FrameData) -> np.ndarray:
    """绘制指标HUD"""
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (280, 200), COLOR_BG, -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    y = 25
    lines = [f"Frame: {fd.frame_idx}  Time: {fd.timestamp:.1f}s"]
    angles = fd.joint_angles
    if angles.right_elbow is not None:
        lines.append(f"R Elbow: {angles.right_elbow:.0f} deg")
    if angles.left_elbow is not None:
        lines.append(f"L Elbow: {angles.left_elbow:.0f} deg")
    if angles.right_knee is not None:
        lines.append(f"R Knee: {angles.right_knee:.0f} deg")
    if angles.left_knee is not None:
        lines.append(f"L Knee: {angles.left_knee:.0f} deg")
    lines.append(f"Wrist Speed: {fd.wrist_speed:.0f} px/f")
    lines.append(f"Body Lean: {fd.body_lean:.0f} deg")

    for line in lines:
        cv2.putText(
            img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA
        )
        y += 25
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
        if progress_callback and i % 10 == 0:
            progress_callback(i / len(frames_data))

    writer.release()
    cap.release()
    return output_path
