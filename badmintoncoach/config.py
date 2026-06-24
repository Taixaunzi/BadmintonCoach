"""配置管理"""
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


class AppConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    upload_dir: str = "./uploads"
    output_dir: str = "./output"


class PoseConfig(BaseModel):
    mode: str = "balanced"
    backend: str = "onnxruntime"
    device: str = "cpu"
    det_frequency: int = 10
    kpt_threshold: float = 0.43


class BallConfig(BaseModel):
    model_path: str = "./models/tracknet.onnx"
    confidence_threshold: float = 0.3


class EventConfig(BaseModel):
    sport: str = "badminton"
    angle_ranges: Dict[str, List[float]] = {
        "left_elbow": [90, 170],
        "right_elbow": [90, 170],
        "left_knee": [100, 170],
        "right_knee": [100, 170],
        "left_shoulder": [20, 120],
        "right_shoulder": [20, 120],
    }
    highlight_thresholds: Dict[str, float] = {
        "wrist_speed_max": 1500,
        "body_lean_max": 35,
        "speed_change_rate": 500,
    }


class SlowmoConfig(BaseModel):
    factor: float = 0.25
    pre_event_frames: int = 30
    post_event_frames: int = 30


class LLMConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4.1-mini"
    temperature: float = 0.3
    max_tokens: int = 500


class Settings(BaseModel):
    app: AppConfig = AppConfig()
    pose: PoseConfig = PoseConfig()
    ball: BallConfig = BallConfig()
    events: EventConfig = EventConfig()
    slowmo: SlowmoConfig = SlowmoConfig()
    llm: LLMConfig = LLMConfig()


def load_config(path: Optional[str] = None) -> Settings:
    p = Path(path) if path else CONFIG_PATH
    if p.exists():
        with open(p) as f:
            data = yaml.safe_load(f) or {}
        return Settings(**data)
    return Settings()
