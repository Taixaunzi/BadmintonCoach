# BadmintonCoach — 系统设计规格

> **日期**：2026-06-24
> **状态**：初稿，待审查
> **项目**：BadmintonCoach — 开源羽毛球视频AI分析系统

---

## 1. 项目概述

**目标**：输入羽毛球运动视频，输出带骨骼+球体标注的视频、运动参数、问题慢镜头、精彩慢镜头、LLM教练评价。

**核心价值**：
- 让业余球友获得专业级的动作分析
- 自动发现技术问题并给出改进建议
- 精彩瞬间自动集锦

## 2. 技术栈

| 层 | 方案 | 版本/说明 |
|---|---|---|
| 后端框架 | FastAPI + Python 3.10+ | REST API + WebSocket |
| 前端框架 | Vue 3 + Vite + TypeScript | SPA |
| 骨骼姿态 | RTMPose (rtmlib) | ONNX推理，无需PyTorch |
| 球体追踪 | TrackNet | 热力图方式，3帧输入 |
| 追踪层 | ByteTrack | 身份关联 |
| 视频处理 | OpenCV + FFmpeg | 读写/标注/慢放 |
| LLM | OpenAI兼容API | 用户自配base_url + api_key |
| 部署 | Docker Compose | 一键启动 |

## 3. 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                      Vue 3 前端                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ 视频上传  │ │ 分析仪表盘│ │ 视频播放器│ │ LLM教练对话  │ │
│  └────┬─────┘ └────▲─────┘ └────▲─────┘ └──────▲───────┘ │
│       │            │            │               │         │
└───────┼────────────┼────────────┼───────────────┼─────────┘
        │ WebSocket  │ REST       │ 文件URL        │ REST
┌───────▼────────────┼────────────┼───────────────┼─────────┐
│                FastAPI 后端                              │
│  ┌──────────┐ ┌────┴────┐ ┌────┴────┐ ┌───────┴───────┐ │
│  │ 上传API  │ │ 状态推送 │ │ 文件服务 │ │ LLM代理API    │ │
│  └────┬─────┘ └─────────┘ └─────────┘ └───────────────┘ │
│       │                                                  │
│  ┌────▼──────────────────────────────────────────────┐   │
│  │              分析引擎 (Pipeline)                    │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────┐ │   │
│  │  │ 帧提取  │→│ 骨骼   │→│ 球追踪 │→│ 参数提取   │ │   │
│  │  │ OpenCV │ │ RTMPose│ │TrackNet│ │ NumPy/SciPy│ │   │
│  │  └────────┘ └────────┘ └────────┘ └─────┬──────┘ │   │
│  │                                          │        │   │
│  │  ┌────────────┐ ┌────────────┐ ┌─────────▼──────┐ │   │
│  │  │ 事件检测   │ │ 视频标注   │ │ 慢镜头生成     │ │   │
│  │  │ 问题/精彩  │ │ 骨骼+球+HUD│ │ 裁剪+慢放+文字 │ │   │
│  │  └────────────┘ └────────────┘ └────────────────┘ │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

## 4. 模块设计

### 4.1 后端模块

```
badmintoncoach/
├── backend/
│   ├── main.py                 # FastAPI入口，路由注册
│   ├── config.py               # 配置管理（YAML加载）
│   ├── api/
│   │   ├── upload.py           # 视频上传API
│   │   ├── analysis.py         # 分析状态查询
│   │   ├── files.py            # 输出文件服务
│   │   └── llm.py              # LLM代理API
│   ├── engine/
│   │   ├── pipeline.py         # 分析管道编排
│   │   ├── pose_estimator.py   # RTMPose封装
│   │   ├── ball_tracker.py     # TrackNet封装
│   │   ├── param_extractor.py  # 关节角/速度/轨迹提取
│   │   ├── event_detector.py   # 问题/精彩事件检测
│   │   ├── annotator.py        # 视频标注（骨架+球+HUD）
│   │   └── slowmo.py           # 慢镜头生成
│   ├── llm/
│   │   ├── client.py           # OpenAI兼容客户端
│   │   ├── prompts.py          # Prompt模板（Skill Router）
│   │   └── formatter.py        # 结构化输出解析
│   └── models/
│       ├── schemas.py          # Pydantic数据模型
│       └── enums.py            # 枚举（事件类型等）
├── frontend/                   # Vue 3 + Vite
│   ├── src/
│   │   ├── App.vue
│   │   ├── views/
│   │   │   ├── UploadView.vue      # 上传页面
│   │   │   ├── AnalysisView.vue    # 分析结果页面
│   │   │   └── SettingsView.vue    # LLM配置页面
│   │   ├── components/
│   │   │   ├── VideoPlayer.vue     # 视频播放器（带时间轴标记）
│   │   │   ├── SkeletonOverlay.vue # 骨骼叠加渲染
│   │   │   ├── MetricsPanel.vue    # 运动指标面板
│   │   │   ├── TimelineMarker.vue  # 时间轴事件标记
│   │   │   ├── SlowmoViewer.vue    # 慢镜头回放
│   │   │   └── CoachChat.vue       # LLM教练对话
│   │   ├── stores/
│   │   │   └── analysis.ts         # Pinia状态管理
│   │   └── api/
│   │       └── index.ts            # API客户端
│   ├── package.json
│   └── vite.config.ts
├── models/                     # 预训练模型（ONNX）
│   ├── rtmpose-m.onnx
│   ├── rtmdet-nano.onnx
│   └── tracknet.onnx
├── docs/                       # 文档
├── tests/                      # 测试
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── config.yaml                 # 默认配置
└── README.md
```

### 4.2 数据模型

```python
# backend/models/schemas.py
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class EventType(str, Enum):
    PROBLEM = "problem"
    HIGHLIGHT = "highlight"
    ACTION = "action"

class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

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
    keypoints: List[List[float]]    # (17, 2)
    scores: List[float]             # (17,)
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
    frames_data: List[FrameData]     # 可选：全部帧数据
    output_files: dict               # {name: path}

class LLMConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4.1-mini"
    temperature: float = 0.3
    max_tokens: int = 500

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

### 4.3 分析管道流程

```python
# backend/engine/pipeline.py
class AnalysisPipeline:
    """分析管道：视频 → 全部输出"""

    def __init__(self, config):
        self.pose_estimator = PoseEstimator(config.pose)
        self.ball_tracker = BallTracker(config.ball)
        self.param_extractor = ParamExtractor()
        self.event_detector = EventDetector(config.events)
        self.annotator = VideoAnnotator()
        self.slowmo_generator = SlowmoGenerator()

    async def run(self, video_path: str, output_dir: str,
                  progress_callback=None) -> AnalysisResult:
        # 1. 帧提取
        frames = self._extract_frames(video_path)

        # 2. 逐帧：骨骼 + 球追踪
        all_frame_data = []
        for i, frame in enumerate(frames):
            kpts, scores = self.pose_estimator(frame)
            ball_pos = self.ball_tracker(frame)
            angles = self.param_extractor.extract(kpts, scores)
            speed = self.param_extractor.calc_wrist_speed(kpts, scores, prev)
            lean = self.param_extractor.calc_body_lean(kpts, scores)
            fd = FrameData(frame_idx=i, keypoints=kpts, scores=scores,
                          ball_position=ball_pos, joint_angles=angles,
                          wrist_speed=speed, body_lean=lean)
            all_frame_data.append(fd)
            if progress_callback:
                progress_callback("pose", i / len(frames))

        # 3. 事件检测
        events = self.event_detector.detect_all(all_frame_data)

        # 4. 标注视频
        full_video = self.annotator.annotate(video_path, all_frame_data, output_dir)

        # 5. 慢镜头生成
        problems_slowmo = self.slowmo_generator.generate(
            video_path, [e for e in events if e.event_type == "problem"], output_dir)
        highlights_slowmo = self.slowmo_generator.generate(
            video_path, [e for e in events if e.event_type == "highlight"], output_dir)

        # 6. 输出汇总
        return AnalysisResult(...)
```

### 4.4 LLM集成

```python
# backend/llm/client.py
from openai import AsyncOpenAI

class CoachLLM:
    """OpenAI兼容的教练LLM客户端"""

    def __init__(self, config: LLMConfig):
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def get_coaching(self, analysis_json: dict) -> str:
        """从分析结果获取教练建议"""
        system = self._build_system_prompt(analysis_json)
        user = self._build_user_prompt(analysis_json)

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content
```

## 5. API设计

### REST API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/upload` | 上传视频，返回video_id |
| GET | `/api/analysis/{id}/status` | 查询分析状态 |
| GET | `/api/analysis/{id}/result` | 获取分析结果（JSON） |
| GET | `/api/analysis/{id}/events` | 获取事件列表 |
| GET | `/api/files/{id}/{filename}` | 下载输出文件 |
| POST | `/api/llm/coach` | LLM教练对话 |
| GET | `/api/config/llm` | 获取LLM配置 |
| PUT | `/api/config/llm` | 更新LLM配置 |

### WebSocket

| 路径 | 说明 |
|------|------|
| `/ws/analysis/{id}` | 实时推送分析进度 |

## 6. 前端页面

### 6.1 上传页面 (`/`)
- 拖拽/点击上传视频
- 支持格式：MP4, MOV, AVI
- 上传后自动开始分析

### 6.2 分析结果页面 (`/analysis/:id`)
- **左侧面板**：视频播放器
  - 骨骼+球轨迹叠加渲染
  - 时间轴标记（红色=问题，橙色=精彩）
  - 点击标记跳转到对应位置
- **右侧面板**：
  - 运动指标仪表盘（关节角度、速度等）
  - 事件列表（可点击展开慢放）
  - LLM教练建议
- **底部**：
  - 问题慢放集锦
  - 精彩瞬间集锦

### 6.3 设置页面 (`/settings`)
- LLM配置（base_url, api_key, model）
- 分析参数调节（检测阈值、慢放倍率等）

## 7. 配置文件

```yaml
# config.yaml
app:
  host: "0.0.0.0"
  port: 8000
  upload_dir: "./uploads"
  output_dir: "./output"

pose:
  mode: "balanced"         # lightweight/balanced/performance
  backend: "onnxruntime"   # onnxruntime/tensorrt/openvino
  device: "cpu"            # cpu/cuda/mps
  det_frequency: 10

ball:
  model_path: "./models/tracknet.onnx"
  confidence_threshold: 0.3

events:
  sport: "badminton"
  angle_ranges:
    elbow: [90, 170]
    knee: [100, 170]
    shoulder: [20, 120]
  highlight_thresholds:
    wrist_speed_max: 1500
    body_lean_max: 35

slowmo:
  factor: 0.25             # 0.25 = 4倍慢放
  pre_event_frames: 30
  post_event_frames: 30

llm:
  base_url: "https://api.openai.com/v1"
  api_key: ""
  model: "gpt-4.1-mini"
  temperature: 0.3
  max_tokens: 500
```

## 8. 部署方案

```yaml
# docker-compose.yml
version: "3.8"
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/app/uploads
      - ./output:/app/output
      - ./models:/app/models
    environment:
      - DEVICE=cpu  # 或 cuda

  frontend:
    build:
      context: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
```

## 9. 测试策略

| 层 | 测试类型 | 覆盖 |
|---|---------|------|
| 关节角度计算 | 单元测试 | angle(), extract_angles() |
| 事件检测 | 单元测试 | 已知数据→预期事件 |
| 球追踪 | 集成测试 | 标注视频→检测结果 |
| API | 集成测试 | 上传→分析→结果 |
| 前端 | E2E | 上传→查看结果 |

## 10. 里程碑

| Phase | 内容 | 产出 |
|-------|------|------|
| 1 | 项目骨架 + 骨骼标注 + 球检测 | 能跑的Demo |
| 2 | 参数提取 + 事件检测 + 慢镜头 | 完整后端 |
| 3 | LLM教练反馈 | 全功能后端 |
| 4 | Vue前端 | 可交互的Web应用 |
| 5 | Docker部署 + 文档 | 可发布 |
