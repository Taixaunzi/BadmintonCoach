# BadmintonCoach 实现计划

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 构建完整的羽毛球视频AI分析系统，包含后端Pipeline + 前端Web界面 + LLM教练反馈

**Architecture:** FastAPI后端（分析引擎 + REST API） + Vue 3前端（视频播放 + 结果展示） + OpenAI兼容LLM

**Tech Stack:** Python 3.10+, FastAPI, rtmlib, TrackNet, OpenCV, Vue 3, Vite, TypeScript

---

## Phase 1: 项目骨架 + 骨骼标注 + 球检测

### Task 1.1: 创建项目基础结构

**Objective:** 建立Python包结构、依赖文件、配置系统

**Files:**
- Create: `badmintoncoach/__init__.py`
- Create: `badmintoncoach/config.py`
- Create: `badmintoncoach/models/__init__.py`
- Create: `badmintoncoach/models/schemas.py`
- Create: `badmintoncoach/models/enums.py`
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `tests/__init__.py`

**Step 1: 创建目录结构**

```bash
cd ~/BadmintonCoach
mkdir -p badmintoncoach/{api,engine,llm,models}
mkdir -p tests frontend models output uploads
```

**Step 2: 写 requirements.txt**

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
pyyaml>=6.0
pydantic>=2.0
numpy>=1.24.0
opencv-python-headless>=4.8.0
rtmlib>=0.0.13
openai>=1.0.0
aiofiles>=23.2.0
scipy>=1.11.0
```

**Step 3: 写 config.yaml**

```yaml
app:
  host: "0.0.0.0"
  port: 8000
  upload_dir: "./uploads"
  output_dir: "./output"

pose:
  mode: "balanced"
  backend: "onnxruntime"
  device: "cpu"
  det_frequency: 10
  kpt_threshold: 0.43

ball:
  model_path: "./models/tracknet.onnx"
  confidence_threshold: 0.3

events:
  sport: "badminton"
  angle_ranges:
    left_elbow: [90, 170]
    right_elbow: [90, 170]
    left_knee: [100, 170]
    right_knee: [100, 170]
    left_shoulder: [20, 120]
    right_shoulder: [20, 120]
  highlight_thresholds:
    wrist_speed_max: 1500
    body_lean_max: 35
    speed_change_rate: 500

slowmo:
  factor: 0.25
  pre_event_frames: 30
  post_event_frames: 30

llm:
  base_url: "https://api.openai.com/v1"
  api_key: ""
  model: "gpt-4.1-mini"
  temperature: 0.3
  max_tokens: 500
```

**Step 4: 写 config.py**

```python
"""配置管理"""
import yaml
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, List, Optional

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
        "left_elbow": [90, 170], "right_elbow": [90, 170],
        "left_knee": [100, 170], "right_knee": [100, 170],
        "left_shoulder": [20, 120], "right_shoulder": [20, 120],
    }
    highlight_thresholds: Dict[str, float] = {
        "wrist_speed_max": 1500, "body_lean_max": 35, "speed_change_rate": 500,
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
```

**Step 5: 写数据模型 schemas.py**

```python
"""Pydantic 数据模型"""
from pydantic import BaseModel
from typing import List, Optional
from .enums import EventType, Severity, AnalysisStatus

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
    keypoints: List[List[float]]
    scores: List[float]
    ball_position: Optional[List[float]] = None
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
    output_files: dict

class AnalysisProgress(BaseModel):
    video_id: str
    status: AnalysisStatus
    progress: float = 0.0
    message: str = ""
```

**Step 6: 写 enums.py**

```python
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
```

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: 项目骨架 — 包结构、配置系统、数据模型

- badmintoncoach/ Python包结构
- config.py: YAML配置加载，Pydantic验证
- models/schemas.py: FrameData, TimelineEvent, AnalysisResult等
- models/enums.py: EventType, Severity, AnalysisStatus枚举
- requirements.txt: 全部依赖
- config.yaml: 默认配置文件"
```

---

### Task 1.2: 关节角度计算模块

**Objective:** 实现从关键点计算关节角度的核心算法

**Files:**
- Create: `badmintoncoach/engine/__init__.py`
- Create: `badmintoncoach/engine/param_extractor.py`
- Create: `tests/test_param_extractor.py`

**Step 1: 写测试**

```python
# tests/test_param_extractor.py
import numpy as np
from badmintoncoach.engine.param_extractor import (
    calculate_angle, extract_joint_angles, calc_wrist_speed, calc_body_lean
)

def test_calculate_angle_right_angle():
    """90度角"""
    a = [0, 0]
    b = [1, 0]
    c = [1, 1]
    assert abs(calculate_angle(a, b, c) - 90.0) < 0.1

def test_calculate_angle_straight():
    """180度（直线）"""
    a = [0, 0]
    b = [1, 0]
    c = [2, 0]
    assert abs(calculate_angle(a, b, c) - 180.0) < 0.1

def test_calculate_angle_acute():
    """锐角"""
    a = [0, 1]
    b = [0, 0]
    c = [1, 1]
    angle = calculate_angle(a, b, c)
    assert 40 < angle < 50

def test_extract_joint_angles_all_visible():
    """所有关键点可见"""
    kpts = np.array([
        [0, 0],    # 0 nose
        [0, 0],    # 1 left_eye
        [0, 0],    # 2 right_eye
        [0, 0],    # 3 left_ear
        [0, 0],    # 4 right_ear
        [-1, 0],   # 5 left_shoulder
        [1, 0],    # 6 right_shoulder
        [-2, 0],   # 7 left_elbow
        [2, 0],    # 8 right_elbow
        [-2, -1],  # 9 left_wrist
        [2, -1],   # 10 right_wrist
        [-1, -2],  # 11 left_hip
        [1, -2],   # 12 right_hip
        [-1, -3],  # 13 left_knee
        [1, -3],   # 14 right_knee
        [-1, -4],  # 15 left_ankle
        [1, -4],   # 16 right_ankle
    ], dtype=float)
    scores = np.ones(17) * 0.9
    angles = extract_joint_angles(kpts, scores)
    assert 'left_elbow' in angles
    assert 'right_knee' in angles

def test_extract_joint_angles_low_confidence():
    """低置信度关键点被忽略"""
    kpts = np.zeros((17, 2))
    scores = np.ones(17) * 0.1  # 低于阈值
    angles = extract_joint_angles(kpts, scores, threshold=0.3)
    assert len(angles) == 0
```

**Step 2: 写实现**

```python
# badmintoncoach/engine/param_extractor.py
"""运动参数提取：关节角度、速度、身体倾斜"""
import numpy as np
from typing import Dict, List, Optional
from ..models.schemas import JointAngles

def calculate_angle(a, b, c) -> float:
    """计算关节b处的角度（度）
    Args:
        a: [x, y] 父关节
        b: [x, y] 当前关节（角度顶点）
        c: [x, y] 子关节
    Returns:
        角度（度）
    """
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

# COCO 17关键点索引
KPT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16,
}

# 关节角度定义：(父关节, 当前关节, 子关节)
ANGLE_DEFS = {
    'left_elbow':    ('left_shoulder', 'left_elbow', 'left_wrist'),
    'right_elbow':   ('right_shoulder', 'right_elbow', 'right_wrist'),
    'left_knee':     ('left_hip', 'left_knee', 'left_ankle'),
    'right_knee':    ('right_hip', 'right_knee', 'right_ankle'),
    'left_shoulder': ('left_elbow', 'left_shoulder', 'left_hip'),
    'right_shoulder':('right_elbow', 'right_shoulder', 'right_hip'),
    'left_hip':      ('left_shoulder', 'left_hip', 'left_knee'),
    'right_hip':     ('right_shoulder', 'right_hip', 'right_knee'),
}

def extract_joint_angles(keypoints: np.ndarray, scores: np.ndarray,
                         threshold: float = 0.3) -> Dict[str, float]:
    """从17个关键点提取8个主要关节角度
    Args:
        keypoints: (17, 2) 关键点坐标
        scores: (17,) 置信度
        threshold: 置信度阈值
    Returns:
        {关节名: 角度(度)} 字典
    """
    angles = {}
    for angle_name, (parent, joint, child) in ANGLE_DEFS.items():
        pi, ji, ci = KPT[parent], KPT[joint], KPT[child]
        if all(scores[i] > threshold for i in [pi, ji, ci]):
            angles[angle_name] = calculate_angle(
                keypoints[pi], keypoints[ji], keypoints[ci])
    return angles

def calc_wrist_speed(keypoints: np.ndarray, scores: np.ndarray,
                     prev_keypoints: Optional[np.ndarray] = None) -> float:
    """计算手腕最大速度（像素/帧）"""
    if prev_keypoints is None:
        return 0.0
    max_speed = 0.0
    for wrist_idx in [9, 10]:  # left_wrist, right_wrist
        if scores[wrist_idx] > 0.3:
            speed = np.linalg.norm(keypoints[wrist_idx] - prev_keypoints[wrist_idx])
            max_speed = max(max_speed, speed)
    return max_speed

def calc_body_lean(keypoints: np.ndarray, scores: np.ndarray) -> float:
    """计算身体倾斜角（脊柱vs垂直方向，度）"""
    # 用右肩-右髋连线代表脊柱
    if scores[6] > 0.3 and scores[12] > 0.3:
        spine = keypoints[6] - keypoints[12]
        vertical = np.array([0, -1], dtype=float)
        norm = np.linalg.norm(spine)
        if norm < 1e-6:
            return 0.0
        cosine = np.dot(spine, vertical) / norm
        return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return 0.0
```

**Step 3: 运行测试**

```bash
cd ~/BadmintonCoach
python -m pytest tests/test_param_extractor.py -v
```

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: 关节角度计算模块 — calculate_angle, extract_joint_angles, calc_wrist_speed, calc_body_lean

- 8个主要关节角度提取（肘/膝/肩/髋，左右对称）
- 手腕速度计算（像素/帧）
- 身体倾斜角计算（脊柱vs垂直方向）
- 完整单元测试覆盖"
```

---

### Task 1.3: 骨骼姿态估计模块

**Objective:** 封装RTMPose为统一接口

**Files:**
- Create: `badmintoncoach/engine/pose_estimator.py`

**实现:**

```python
# badmintoncoach/engine/pose_estimator.py
"""骨骼姿态估计 — 基于rtmlib的RTMPose封装"""
import numpy as np
from typing import Tuple, Optional
from ..config import PoseConfig

class PoseEstimator:
    """RTMPose姿态估计器"""

    def __init__(self, config: PoseConfig):
        self.config = config
        self._tracker = None

    def _init_tracker(self):
        if self._tracker is not None:
            return
        from rtmlib import Body, PoseTracker
        self._tracker = PoseTracker(
            Body,
            mode=self.config.mode,
            det_frequency=self.config.det_frequency,
            backend=self.config.backend,
            device=self.config.device,
        )

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检测单帧姿态
        Args:
            frame: BGR图像 (H, W, 3)
        Returns:
            keypoints: (N, 17, 2) 关键点坐标
            scores: (N, 17) 置信度
        """
        self._init_tracker()
        keypoints, scores = self._tracker(frame)
        return keypoints, scores

    def reset(self):
        """重置追踪器状态（处理新视频时调用）"""
        self._tracker = None
```

**Commit:**

```bash
git add -A
git commit -m "feat: 骨骼姿态估计模块 — RTMPose封装，支持balanced/lightweight/performance模式"
```

---

### Task 1.4: 球体追踪模块

**Objective:** 封装TrackNet为统一接口（先用占位实现，后续集成真实模型）

**Files:**
- Create: `badmintoncoach/engine/ball_tracker.py`

**实现:**

```python
# badmintoncoach/engine/ball_tracker.py
"""球体追踪 — TrackNet热力图检测"""
import numpy as np
from typing import Optional
from ..config import BallConfig

class BallTracker:
    """羽毛球追踪器"""

    def __init__(self, config: BallConfig):
        self.config = config
        self._model = None
        self._frame_buffer = []  # TrackNet需要3帧输入

    def __call__(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """检测单帧球位置
        Args:
            frame: BGR图像 (H, W, 3)
        Returns:
            ball_position: [x, y] 或 None（未检测到）
        """
        import cv2
        # 缓冲3帧
        resized = cv2.resize(frame, (640, 360))
        self._frame_buffer.append(resized)
        if len(self._frame_buffer) > 3:
            self._frame_buffer.pop(0)

        if len(self._frame_buffer) < 3:
            return None

        # TODO: 集成真实TrackNet模型
        # 目前用简单的颜色检测作为占位
        return self._detect_by_color(frame)

    def _detect_by_color(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """基于颜色的简单球体检测（占位实现）"""
        import cv2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 羽毛球通常是白色/黄色
        lower = np.array([15, 50, 200])
        upper = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 20:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return np.array([cx, cy], dtype=float)
        return None

    def reset(self):
        self._frame_buffer = []
```

**Commit:**

```bash
git add -A
git commit -m "feat: 球体追踪模块 — TrackNet框架+颜色检测占位实现

- 3帧缓冲（为TrackNet准备）
- 颜色检测作为临时占位
- 后续替换为真实TrackNet ONNX模型"
```

---

### Task 1.5: 视频标注模块

**Objective:** 在视频帧上绘制骨骼+球+指标HUD

**Files:**
- Create: `badmintoncoach/engine/annotator.py`

**实现:**

```python
# badmintoncoach/engine/annotator.py
"""视频标注：骨骼 + 球 + 指标HUD"""
import cv2
import numpy as np
from typing import List, Optional
from ..models.schemas import FrameData

# COCO骨架连接
SKELETON = [
    (5, 7), (7, 9),     # 左肩-左肘-左腕
    (6, 8), (8, 10),    # 右肩-右肘-右腕
    (5, 6),             # 左肩-右肩
    (5, 11), (6, 12),   # 肩-髋
    (11, 13), (13, 15), # 左髋-左膝-左踝
    (12, 14), (14, 16), # 右髋-右膝-右踝
    (11, 12),           # 左髋-右髋
]

# 颜色定义
COLOR_BONE = (0, 255, 0)      # 绿色骨架
COLOR_KPT = (0, 0, 255)       # 红色关键点
COLOR_BALL = (0, 165, 255)    # 橙色球
COLOR_TEXT = (255, 255, 255)  # 白色文字
COLOR_BG = (0, 0, 0)          # 黑色背景

def annotate_frame(frame: np.ndarray, frame_data: FrameData,
                   kpt_threshold: float = 0.43) -> np.ndarray:
    """在单帧上绘制标注
    Args:
        frame: BGR图像
        frame_data: 该帧的分析数据
        kpt_threshold: 关键点置信度阈值
    Returns:
        标注后的图像
    """
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

    # 半透明背景条
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
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, COLOR_TEXT, 1, cv2.LINE_AA)
        y += 25

    return img

def annotate_video(input_path: str, output_path: str,
                   frames_data: List[FrameData],
                   progress_callback=None) -> str:
    """标注完整视频
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        frames_data: 每帧的分析数据
    Returns:
        输出视频路径
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i, fd in enumerate(frames_data):
        ret, frame = cap.read()
        if not ret:
            break
        annotated = annotate_frame(frame, fd)
        writer.write(annotated)
        if progress_callback:
            progress_callback(i / len(frames_data))

    writer.release()
    cap.release()
    return output_path
```

**Commit:**

```bash
git add -A
git commit -m "feat: 视频标注模块 — 骨骼+球+指标HUD绘制

- 骨骼连线（绿色）+ 关键点（红色）+ 球（橙色）
- HUD面板：帧号/时间/关节角度/手腕速度/身体倾斜
- 完整视频标注：逐帧处理+进度回调"
```

---

### Task 1.6: FastAPI服务器 + 上传API

**Objective:** 创建FastAPI入口和视频上传接口

**Files:**
- Create: `badmintoncoach/server.py`
- Create: `badmintoncoach/api/__init__.py`
- Create: `badmintoncoach/api/upload.py`
- Create: `badmintoncoach/api/files.py`

**Step 1: server.py**

```python
# badmintoncoach/server.py
"""FastAPI 服务器入口"""
import uuid
import asyncio
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .config import load_config
from .api.upload import router as upload_router
from .api.files import router as files_router

config = load_config()

app = FastAPI(title="BadmintonCoach", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 确保目录存在
Path(config.app.upload_dir).mkdir(parents=True, exist_ok=True)
Path(config.app.output_dir).mkdir(parents=True, exist_ok=True)

# 静态文件（输出视频）
app.mount("/output", StaticFiles(directory=config.app.output_dir), name="output")

app.include_router(upload_router, prefix="/api")
app.include_router(files_router, prefix="/api")

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("badmintoncoach.server:app",
                host=config.app.host, port=config.app.port, reload=True)
```

**Step 2: upload.py**

```python
# badmintoncoach/api/upload.py
"""视频上传API"""
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from ..config import load_config

router = APIRouter()
config = load_config()

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """上传视频，返回video_id"""
    # 验证文件类型
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的格式: {ext}，支持: {ALLOWED_EXTENSIONS}")

    # 生成video_id
    video_id = str(uuid.uuid4())[:8]
    upload_dir = Path(config.app.upload_dir) / video_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # 保存文件
    file_path = upload_dir / f"input{ext}"
    with open(file_path, "wb") as f:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            shutil.rmtree(upload_dir)
            raise HTTPException(400, "文件过大（最大500MB）")
        f.write(content)

    return {
        "video_id": video_id,
        "filename": file.filename,
        "size": len(content),
        "path": str(file_path),
    }
```

**Step 3: files.py**

```python
# badmintoncoach/api/files.py
"""文件服务API"""
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from ..config import load_config

router = APIRouter()
config = load_config()

@router.get("/files/{video_id}/{filename}")
async def get_file(video_id: str, filename: str):
    """下载输出文件"""
    file_path = Path(config.app.output_dir) / video_id / filename
    if not file_path.exists():
        raise HTTPException(404, "文件不存在")
    return FileResponse(file_path)
```

**Step 4: 运行测试**

```bash
cd ~/BadmintonCoach
pip install -e .
python -c "from badmintoncoach.server import app; print('OK')"
```

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: FastAPI服务器 — 视频上传API + 文件服务

- POST /api/upload: 视频上传，返回video_id
- GET /api/files/{id}/{name}: 输出文件下载
- CORS允许跨域
- 文件大小/类型验证"
```

---

### Task 1.7: 分析管道编排

**Objective:** 将所有模块串联成完整的分析管道

**Files:**
- Create: `badmintoncoach/engine/pipeline.py`
- Create: `badmintoncoach/engine/event_detector.py`
- Create: `badmintoncoach/engine/slowmo.py`

**pipeline.py:**

```python
# badmintoncoach/engine/pipeline.py
"""分析管道：视频 → 全部输出"""
import json
import cv2
from pathlib import Path
from typing import Callable, Optional
from ..config import Settings
from ..models.schemas import FrameData, JointAngles, AnalysisResult, AnalysisProgress
from ..models.enums import AnalysisStatus
from .pose_estimator import PoseEstimator
from .ball_tracker import BallTracker
from .param_extractor import extract_joint_angles, calc_wrist_speed, calc_body_lean
from .event_detector import EventDetector
from .annotator import annotate_video
from .slowmo import generate_slowmo_clips
import numpy as np

class AnalysisPipeline:
    def __init__(self, config: Settings):
        self.config = config
        self.pose = PoseEstimator(config.pose)
        self.ball = BallTracker(config.ball)
        self.event_detector = EventDetector(config.events, config.slowmo)

    def run(self, video_path: str, output_dir: str,
            progress_callback: Optional[Callable] = None) -> AnalysisResult:
        """运行完整分析管道"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps > 0 else 0

        def notify(status, prog, msg=""):
            if progress_callback:
                progress_callback(AnalysisProgress(
                    video_id="", status=status, progress=prog, message=msg))

        # Step 1: 逐帧分析
        notify(AnalysisStatus.POSE, 0.0, "骨骼姿态估计中...")
        frames_data = []
        prev_kpts = None

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
                frame_idx=i, timestamp=i / fps,
                keypoints=kpt.tolist(), scores=sc.tolist(),
                ball_position=ball_pos.tolist() if ball_pos is not None else None,
                joint_angles=JointAngles(**angles),
                wrist_speed=speed, body_lean=lean,
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
            video_path, events, output_dir, self.config.slowmo)

        # Step 5: 保存JSON
        report_path = str(Path(output_dir) / "report.json")
        report = {
            "total_frames": len(frames_data),
            "fps": fps,
            "duration": duration,
            "events": [e.model_dump() for e in events],
            "frames_summary": {
                "avg_wrist_speed": np.mean([f.wrist_speed for f in frames_data]),
                "avg_body_lean": np.mean([f.body_lean for f in frames_data]),
            }
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        notify(AnalysisStatus.DONE, 1.0, "分析完成!")

        return AnalysisResult(
            video_id=Path(output_dir).name,
            total_frames=len(frames_data), fps=fps, duration=duration,
            events=events,
            output_files={
                "full_analysis": full_path,
                "problems_slowmo": problem_path,
                "highlights_slowmo": highlight_path,
                "report": report_path,
            },
        )
```

**event_detector.py:**

```python
# badmintoncoach/engine/event_detector.py
"""事件检测：问题 + 精彩瞬间"""
from typing import List
from ..models.schemas import FrameData, TimelineEvent
from ..models.enums import EventType, Severity
from ..config import EventConfig, SlowmoConfig

class EventDetector:
    def __init__(self, event_config: EventConfig, slowmo_config: SlowmoConfig):
        self.angle_ranges = event_config.angle_ranges
        self.thresholds = event_config.highlight_thresholds
        self.pre_frames = slowmo_config.pre_event_frames
        self.post_frames = slowmo_config.post_event_frames

    def detect_all(self, frames: List[FrameData]) -> List[TimelineEvent]:
        events = []
        for i, fd in enumerate(frames):
            events.extend(self._detect_frame(fd, frames, i))
        return self._merge_events(events)

    def _detect_frame(self, fd: FrameData, all_frames, idx) -> List[TimelineEvent]:
        events = []
        angles = fd.joint_angles.model_dump()

        # 问题检测
        for joint, (lo, hi) in self.angle_ranges.items():
            val = angles.get(joint)
            if val is None:
                continue
            if val < lo:
                events.append(TimelineEvent(
                    frame_idx=fd.frame_idx, timestamp=fd.timestamp,
                    event_type=EventType.PROBLEM,
                    sub_type=f"{joint}_too_closed",
                    severity=Severity.WARNING if val > lo * 0.8 else Severity.CRITICAL,
                    description=f"{joint}角度过小: {val:.0f}°（正常{lo}-{hi}°）",
                    improvement=f"注意{joint}不要过度弯曲，保持{lo}°以上",
                    start_frame=max(0, fd.frame_idx - self.pre_frames),
                    end_frame=fd.frame_idx + self.post_frames,
                    score=abs(val - lo) / lo * 100,
                ))
            elif val > hi:
                events.append(TimelineEvent(
                    frame_idx=fd.frame_idx, timestamp=fd.timestamp,
                    event_type=EventType.PROBLEM,
                    sub_type=f"{joint}_overextend",
                    severity=Severity.WARNING if val < hi * 1.1 else Severity.CRITICAL,
                    description=f"{joint}角度过大: {val:.0f}°（正常{lo}-{hi}°）",
                    improvement=f"注意{joint}不要过度伸展，保持{hi}°以内",
                    start_frame=max(0, fd.frame_idx - self.pre_frames),
                    end_frame=fd.frame_idx + self.post_frames,
                    score=abs(val - hi) / hi * 100,
                ))

        # 精彩瞬间
        if fd.wrist_speed > self.thresholds["wrist_speed_max"]:
            events.append(TimelineEvent(
                frame_idx=fd.frame_idx, timestamp=fd.timestamp,
                event_type=EventType.HIGHLIGHT,
                sub_type="fast_swing",
                severity=Severity.INFO,
                description=f"高速挥拍！手腕速度 {fd.wrist_speed:.0f}",
                improvement="",
                start_frame=max(0, fd.frame_idx - self.pre_frames),
                end_frame=fd.frame_idx + self.post_frames,
                score=min(100, fd.wrist_speed / 20),
            ))

        if fd.body_lean > self.thresholds["body_lean_max"]:
            events.append(TimelineEvent(
                frame_idx=fd.frame_idx, timestamp=fd.timestamp,
                event_type=EventType.HIGHLIGHT,
                sub_type="extreme_lean",
                severity=Severity.INFO,
                description=f"极限救球！身体倾斜 {fd.body_lean:.0f}°",
                improvement="",
                start_frame=max(0, fd.frame_idx - self.pre_frames),
                end_frame=fd.frame_idx + self.post_frames,
                score=min(100, fd.body_lean * 2),
            ))

        return events

    def _merge_events(self, events: List[TimelineEvent],
                      gap: int = 60) -> List[TimelineEvent]:
        if not events:
            return []
        events.sort(key=lambda e: e.start_frame)
        merged = [events[0]]
        for ev in events[1:]:
            last = merged[-1]
            if (ev.event_type == last.event_type and
                ev.start_frame <= last.end_frame + gap):
                last.end_frame = max(last.end_frame, ev.end_frame)
                last.score = max(last.score, ev.score)
            else:
                merged.append(ev)
        return merged
```

**slowmo.py:**

```python
# badmintoncoach/engine/slowmo.py
"""慢镜头生成"""
import cv2
import numpy as np
from typing import List
from ..models.schemas import TimelineEvent
from ..config import SlowmoConfig

def generate_slowmo_clips(video_path: str, events: List[TimelineEvent],
                          output_dir: str, config: SlowmoConfig):
    """生成问题集锦和精彩集锦"""
    problem_events = [e for e in events if e.event_type.value == "problem"]
    highlight_events = [e for e in events if e.event_type.value == "highlight"]

    problem_path = f"{output_dir}/problems_slowmo.mp4"
    highlight_path = f"{output_dir}/highlights_slowmo.mp4"

    _concat_slowmo(video_path, problem_events, problem_path, config)
    _concat_slowmo(video_path, highlight_events, highlight_path, config)

    return problem_path, highlight_path

def _concat_slowmo(video_path: str, events: List[TimelineEvent],
                   output_path: str, config: SlowmoConfig):
    if not events:
        # 无事件，创建空视频
        cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480)).release()
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = fps * config.factor

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (w, h))

    events.sort(key=lambda e: e.start_frame)

    for ev in events:
        # 标题卡
        title = np.zeros((h, w, 3), dtype=np.uint8)
        color = (0, 0, 255) if ev.event_type.value == "problem" else (0, 200, 255)
        label = "问题" if ev.event_type.value == "problem" else "精彩"
        cv2.putText(title, f"{label}: {ev.description[:30]}", (w // 6, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        for _ in range(int(out_fps * 1.5)):
            writer.write(title)

        # 慢放片段
        cap.set(cv2.CAP_PROP_POS_FRAMES, ev.start_frame)
        for i in range(ev.start_frame, ev.end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            # 叠加文字
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
            cv2.putText(frame, ev.description[:60], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if ev.improvement:
                cv2.putText(frame, f"改进: {ev.improvement[:50]}", (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            writer.write(frame)

    writer.release()
    cap.release()
```

**Commit:**

```bash
git add -A
git commit -m "feat: 分析管道 — 串联骨骼/球/参数/事件/标注/慢放

- pipeline.py: 完整分析编排，进度回调
- event_detector.py: 关节角度超标检测 + 精彩瞬间检测
- slowmo.py: 问题集锦 + 精彩集锦慢放生成
- 输出：标注视频 + 慢放视频 + JSON报告"
```

---

### Task 1.8: 端到端验证

**Objective:** 用真实视频验证完整管道

**Step 1: 准备测试视频**

```bash
# 用ffmpeg生成一个10秒的测试视频（如果无真实视频）
ffmpeg -f lavfi -i testsrc=duration=10:size=1280x720:rate=30 -pix_fmt yuv420p ~/BadmintonCoach/tests/test_video.mp4
```

**Step 2: 运行分析**

```bash
cd ~/BadmintonCoach
python -c "
from badmintoncoach.config import load_config
from badmintoncoach.engine.pipeline import AnalysisPipeline

config = load_config()
pipeline = AnalysisPipeline(config)
result = pipeline.run('tests/test_video.mp4', 'output/test_run')
print(f'Frames: {result.total_frames}')
print(f'Events: {len(result.events)}')
print(f'Files: {result.output_files}')
"
```

**Step 3: 验证输出文件存在**

```bash
ls -la output/test_run/
```

**Step 4: Commit**

```bash
git add -A
git commit -m "test: 端到端验证 — 测试视频分析管道完整性"
```

---

## Phase 2: LLM教练反馈（续）

*Phase 2将在Phase 1完成后编写，包含LLM集成、前端开发、Docker部署。*

---

## 里程碑

| Phase | 内容 | 预计任务数 |
|-------|------|-----------|
| 1 | 项目骨架 + CV Pipeline | 8个Task |
| 2 | LLM集成 | 3个Task |
| 3 | Vue前端 | 5个Task |
| 4 | Docker部署 + 文档 | 2个Task |
