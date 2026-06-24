"""枚举定义"""
from enum import Enum


class EventType(str, Enum):
    PROBLEM = "problem"
    HIGHLIGHT = "highlight"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting_frames"
    POSE = "pose_estimation"
    BALL = "ball_tracking"
    PARAMS = "param_extraction"
    EVENTS = "event_detection"
    ANNOTATING = "annotating"
    SLOWMO = "generating_slowmo"
    LLM = "llm_coaching"
    DONE = "done"
    ERROR = "error"
