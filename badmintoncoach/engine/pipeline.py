"""分析管道：视频 → 全部输出"""
import json
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from ..config import Settings
from ..models.enums import AnalysisStatus
from ..models.schemas import AnalysisProgress, AnalysisResult, FrameData, JointAngles
from .annotator import annotate_video
from .ball_tracker import BallTracker
from .event_detector import EventDetector
from .param_extractor import calc_body_lean, calc_wrist_speed, extract_joint_angles
from .pose_estimator import PoseEstimator
from .slowmo import generate_slowmo_clips


class AnalysisPipeline:
    """完整分析管道"""

    def __init__(self, config: Settings):
        self.config = config
        self.pose = PoseEstimator(config.pose)
        self.ball = BallTracker(config.ball)
        self.event_detector = EventDetector(config.events, config.slowmo)

    def run(
        self,
        video_path: str,
        output_dir: str,
        video_id: str = "",
        progress_callback: Optional[Callable] = None,
    ) -> AnalysisResult:
        """运行完整分析管道"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # 默认30fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps

        def notify(status: AnalysisStatus, prog: float, msg: str = ""):
            if progress_callback:
                progress_callback(
                    AnalysisProgress(
                        video_id=video_id, status=status, progress=prog, message=msg
                    )
                )

        # Step 1: 逐帧分析（骨骼 + 球 + 参数）
        notify(AnalysisStatus.POSE, 0.0, "骨骼姿态估计中...")
        frames_data: list[FrameData] = []
        prev_kpts: Optional[np.ndarray] = None

        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            # 骨骼
            kpts, scores = self.pose(frame)
            kpt = kpts[0] if len(kpts) > 0 else np.zeros((17, 2))
            sc = scores[0] if len(scores) > 0 else np.zeros(17)

            # 球
            ball_pos = self.ball(frame)

            # 参数
            angles = extract_joint_angles(kpt, sc)
            speed = calc_wrist_speed(kpt, sc, prev_kpts)
            lean = calc_body_lean(kpt, sc)

            fd = FrameData(
                frame_idx=i,
                timestamp=i / fps,
                keypoints=kpt.tolist(),
                scores=sc.tolist(),
                ball_position=ball_pos.tolist() if ball_pos is not None else None,
                joint_angles=JointAngles(**angles),
                wrist_speed=speed,
                body_lean=lean,
            )
            frames_data.append(fd)
            prev_kpts = kpt.copy()

            if i % 30 == 0:
                prog = i / total * 0.5
                notify(AnalysisStatus.POSE, prog, f"骨骼分析 {i}/{total}")

        cap.release()

        # Step 2: 事件检测
        notify(AnalysisStatus.EVENTS, 0.5, "事件检测中...")
        events = self.event_detector.detect_all(frames_data)

        # Step 3: 标注视频
        notify(AnalysisStatus.ANNOTATING, 0.6, "生成标注视频...")
        full_path = str(Path(output_dir) / "full_analysis.mp4")
        annotate_video(video_path, full_path, frames_data)

        # Step 4: 慢镜头
        notify(AnalysisStatus.SLOWMO, 0.8, "生成慢镜头...")
        problem_path, highlight_path = generate_slowmo_clips(
            video_path, events, output_dir, self.config.slowmo
        )

        # Step 5: 保存JSON报告
        report_path = str(Path(output_dir) / "report.json")
        report = {
            "total_frames": len(frames_data),
            "fps": fps,
            "duration": duration,
            "events": [e.model_dump() for e in events],
            "frames_summary": {
                "avg_wrist_speed": float(np.mean([f.wrist_speed for f in frames_data])),
                "avg_body_lean": float(np.mean([f.body_lean for f in frames_data])),
            },
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        notify(AnalysisStatus.DONE, 1.0, "分析完成!")

        return AnalysisResult(
            video_id=video_id or Path(output_dir).name,
            total_frames=len(frames_data),
            fps=fps,
            duration=duration,
            events=events,
            output_files={
                "full_analysis": full_path,
                "problems_slowmo": problem_path,
                "highlights_slowmo": highlight_path,
                "report": report_path,
            },
        )
