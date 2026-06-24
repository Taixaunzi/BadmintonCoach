# 🏸 BadmintonCoach

开源羽毛球视频AI分析系统 — 骨骼标定 + 球体追踪 + 运动参数 + LLM教练反馈

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![Vue 3](https://img.shields.io/badge/Vue-3-brightgreen.svg)](https://vuejs.org)

## ✨ 功能

| 功能 | 说明 | 技术 |
|------|------|------|
| 🦴 骨骼标定 | 17关键点实时标注 | RTMPose (rtmlib) |
| 🏸 球体追踪 | 羽毛球检测+轨迹+速度 | TrackNet 深度学习 |
| 📊 运动参数 | 关节角度、手腕速度、身体倾斜 | NumPy + SciPy |
| ⚠️ 问题检测 | 关节角度超标、动作偏差 | 滑窗聚合+状态机 |
| 🌟 精彩捕捉 | 高速挥拍、极限救球 | 多条件联合判定 |
| 🎬 慢镜头回放 | 问题/精彩片段4倍慢放 | OpenCV |
| 🤖 AI教练 | OpenAI兼容API教练建议 | LLM (任意兼容API) |
| 🌐 Web界面 | 拖拽上传+结果浏览+LLM对话 | Vue 3 + FastAPI |
| 🎾 运动员过滤 | 球场检测+球拍识别+身高评分 | HSV+形状分析 |
| ⚡ GPU加速 | CUDA/MPS/ONNX/TensorRT | PyTorch/ONNX Runtime |

## 🏗️ 技术栈

```
后端:   FastAPI + Python 3.10+
前端:   Vue 3 + Vite + TypeScript + Pinia
骨骼:   RTMPose (rtmlib, ONNX Runtime)
球追踪: TrackNet (PyTorch/ONNX, 3帧热力图)
视频:   OpenCV + PIL (中文渲染)
LLM:    OpenAI兼容API (用户自配)
部署:   Docker Compose
```

## 🚀 快速开始

### Docker Compose（推荐）

```bash
git clone https://github.com/Taixaunzi/BadmintonCoach.git
cd BadmintonCoach

# 配置LLM（可选）
vim config.yaml  # 设置 llm.api_key

# 一键启动
docker-compose up -d
# 访问 http://localhost:3000
```

### 本地开发

```bash
# 后端
pip install -r requirements.txt
python -m uvicorn badmintoncoach.server:app --reload --port 8000

# 前端
cd frontend && npm install && npm run dev

# 访问 http://localhost:3000
```

### GPU加速

```bash
# 检测GPU
python -m badmintoncoach.tools.gpu_utils

# NVIDIA GPU加速
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu

# config.yaml
# pose.device: cuda
# ball.device: cuda
```

## 📁 项目结构

```
BadmintonCoach/
├── badmintoncoach/             # Python后端
│   ├── server.py               # FastAPI入口
│   ├── config.py               # 配置管理 (YAML)
│   ├── api/                    # REST API
│   │   ├── upload.py           # 视频上传
│   │   ├── analysis.py         # 分析触发/状态/结果
│   │   ├── files.py            # 文件下载
│   │   └── ws.py               # WebSocket进度
│   ├── engine/                 # 分析引擎
│   │   ├── pipeline.py         # 分析管道编排
│   │   ├── pose_estimator.py   # RTMPose + 运动员过滤
│   │   ├── ball_tracker.py     # TrackNet球追踪
│   │   ├── racket_detector.py  # 球拍检测
│   │   ├── param_extractor.py  # 关节角度/速度
│   │   ├── event_detector.py   # 滑窗+状态机事件检测
│   │   ├── annotator.py        # 视频标注 (中文HUD)
│   │   ├── slowmo.py           # 慢镜头生成
│   │   └── tracknet_model.py   # TrackNet模型定义
│   ├── llm/                    # LLM集成
│   │   ├── client.py           # OpenAI兼容客户端
│   │   └── prompts.py          # Skill Router Prompt
│   ├── models/                 # 数据模型
│   │   ├── schemas.py          # Pydantic schemas
│   │   └── enums.py            # 枚举定义
│   └── tools/                  # 工具
│       └── gpu_utils.py        # GPU检测+ONNX导出
├── frontend/                   # Vue 3 前端
│   └── src/
│       ├── views/              # 上传/分析/设置页面
│       ├── stores/             # Pinia状态管理
│       └── api/                # API客户端
├── models/                     # 预训练模型
│   ├── tracknet_best.pth       # TrackNet PyTorch权重
│   └── tracknet_best.onnx      # TrackNet ONNX模型
├── tests/                      # 单元测试
├── docs/                       # 设计文档
├── config.yaml                 # 配置文件
├── Dockerfile                  # 后端镜像
├── Dockerfile.frontend         # 前端镜像
├── docker-compose.yml          # 编排
├── nginx.conf                  # Nginx反代
├── requirements.txt            # Python依赖
└── LICENSE                     # MIT
```

## ⚙️ 配置

```yaml
# config.yaml
pose:
  mode: "balanced"           # lightweight / balanced / performance
  backend: "onnxruntime"     # onnxruntime / tensorrt
  device: "cpu"              # cpu / cuda / mps

ball:
  model_path: "./models/tracknet_best.pth"
  device: "auto"             # auto / cpu / cuda / mps

events:
  sport: "badminton"
  angle_ranges:              # 关节角度正常范围
    left_elbow: [15, 175]
    right_elbow: [15, 175]
    left_knee: [50, 175]
    right_knee: [50, 175]

llm:
  base_url: "https://api.openai.com/v1"
  api_key: "sk-..."
  model: "gpt-4.1-mini"
```

## 📊 分析输出

| 文件 | 说明 |
|------|------|
| `full_analysis.mp4` | 标注视频（骨骼+球轨迹+速度+中文HUD） |
| `problems_slowmo.mp4` | 问题片段4倍慢放+时间戳 |
| `highlights_slowmo.mp4` | 精彩瞬间4倍慢放+时间戳 |
| `report.json` | 结构化运动参数+事件列表 |
| `coaching.md` | LLM教练建议 |

## 🔌 API端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/upload` | 上传视频 |
| POST | `/api/analysis/{id}` | 触发分析 |
| GET | `/api/analysis/{id}/status` | 查询状态 |
| GET | `/api/analysis/{id}/result` | 获取结果+LLM建议 |
| GET | `/api/analysis/{id}/events` | 获取事件列表 |
| POST | `/api/llm/chat` | LLM自由对话 |
| GET | `/api/config/llm` | 获取LLM配置 |
| WS | `/ws/analysis/{id}` | 实时进度推送 |

## 🧪 测试

```bash
python -m pytest tests/ -v
```

## 📹 视频要求

| 要求 | 说明 |
|------|------|
| ✅ 近距离拍摄 | 运动员身高占画面>25% |
| ✅ 固定机位 | 球场边固定摄像头 |
| ✅ 光线充足 | 球场灯光均匀 |
| ❌ 广播远景 | 运动员太小，骨架不准 |
| ❌ 频繁切换 | 镜头切换导致追踪丢失 |

## 🤝 贡献

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📄 License

MIT License - 详见 [LICENSE](LICENSE)

## 🙏 致谢

- [MMPose/rtmlib](https://github.com/open-mmlab/mmpose) - RTMPose姿态估计
- [TrackNet](https://github.com/yastrebksv/TrackNet) - 羽毛球追踪模型
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO系列
- [Talking Tennis](https://arxiv.org/abs/2510.03921) - LLM运动教练反馈研究
