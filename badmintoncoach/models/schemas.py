"""Pydantic 数据模型"""
from pydantic import BaseModel
from typing import Dict, List, Optional

from .enums import AnalysisStatus, EventType, Severity


class JointAngles(BaseModel):
    left_elbow: Optional[float] = None
    right_elbow: Optional[float] = None
    left_knee: Optional[float] = None
    right_knee: Optional[float] = None
    left_shoulder: Optional[float] = None
    right_shoulder: Optional[float] = None
    left_hip: Optional[float] = None
    right_hip: Optional[float] = None


class FrameData(BaseModel):
    frame_idx: int
    timestamp: float
    keypoints: List[List[float]]  # (17, 2)
    scores: List[float]  # (17,)
    ball_position: Optional[List[float]] = None  # [x, y]
    joint_angles: JointAngles
    wrist_speed: float = 0.0
    body_lean: float = 0.0


class TimelineEvent(BaseModel):
    frame_idx: int
    timestamp: float
    event_type: EventType
    sub_type: str
    severity: Severity
    description: str
    improvement: str
    start_frame: int
    end_frame: int
    score: float = 0.0


class AnalysisResult(BaseModel):
    video_id: str
    total_frames: int
    fps: float
    duration: float
    events: List[TimelineEvent]
    output_files: Dict[str, str]


class AnalysisProgress(BaseModel):
    video_id: str
    status: AnalysisStatus
    progress: float = 0.0
    message: str = ""
