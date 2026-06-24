# 🏸 BadmintonCoach

开源羽毛球视频AI分析系统 — 骨骼标定 + 球体追踪 + 运动参数 + LLM教练反馈

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![Vue 3](https://img.shields.io/badge/Vue-3-brightgreen.svg)](https://vuejs.org)

## ✨ 功能

| 功能 | 说明 |
|------|------|
| 🦴 骨骼标定 | RTMPose 实时标注17个关键点 |
| 🏸 球体追踪 | TrackNet 热力图追踪羽毛球（亚像素精度） |
| 📊 运动参数 | 关节角度、手腕速度、身体倾斜 |
| ⚠️ 问题检测 | 关节角度超标、动作偏差自动标记 |
| 🌟 精彩捕捉 | 高速挥拍、极限救球自动识别 |
| 🎬 慢放回放 | 问题/精彩片段 4倍慢放 + 文字说明 |
| 🤖 AI教练 | OpenAI兼容API，支持任意LLM |
| 🌐 Web界面 | Vue 3 深色主题，拖拽上传 + 结果浏览 |

## 🏗️ 技术栈

```
后端: FastAPI + Python 3.10+
前端: Vue 3 + Vite + TypeScript + Pinia
骨骼: RTMPose (rtmlib, ONNX)
球追踪: TrackNet (热力图, 3帧输入)
视频: OpenCV
LLM: OpenAI兼容API（用户自配）
部署: Docker Compose
```

## 🚀 快速开始

### 方式一：Docker Compose（推荐）

```bash
git clone https://github.com/Taixaunzi/BadmintonCoach.git
cd BadmintonCoach

# 配置LLM（可选）
vim config.yaml  # 设置 llm.api_key

# 一键启动
docker-compose up -d

# 访问 http://localhost:3000
```

### 方式二：本地开发

```bash
# 后端
pip install -r requirements.txt
python -m uvicorn badmintoncoach.server:app --reload --port 8000

# 前端（另一个终端）
cd frontend
npm install
npm run dev

# 访问 http://localhost:3000
```

## 📁 项目结构

```
BadmintonCoach/
├── badmintoncoach/             # Python后端包
│   ├── server.py               # FastAPI入口
│   ├── config.py               # 配置管理
│   ├── api/                    # REST API路由
│   │   ├── upload.py           # 视频上传
│   │   ├── analysis.py         # 分析触发/状态/结果
│   │   ├── files.py            # 文件下载
│   │   └── ws.py               # WebSocket进度
│   ├── engine/                 # 分析引擎
│   │   ├── pipeline.py         # 分析管道编排
│   │   ├── pose_estimator.py   # RTMPose封装
│   │   ├── ball_tracker.py     # TrackNet封装
│   │   ├── param_extractor.py  # 关节角度/速度
│   │   ├── event_detector.py   # 问题/精彩检测
│   │   ├── annotator.py        # 视频标注
│   │   └── slowmo.py           # 慢镜头生成
│   ├── llm/                    # LLM集成
│   │   ├── client.py           # OpenAI兼容客户端
│   │   └── prompts.py          # Prompt模板
│   └── models/                 # 数据模型
│       ├── schemas.py          # Pydantic schemas
│       └── enums.py            # 枚举定义
├── frontend/                   # Vue 3 前端
│   ├── src/
│   │   ├── views/              # 页面组件
│   │   ├── components/         # 通用组件
│   │   ├── stores/             # Pinia状态
│   │   └── api/                # API客户端
│   └── vite.config.ts
├── models/                     # 预训练模型（ONNX）
├── tests/                      # 测试
├── docs/                       # 设计文档
├── config.yaml                 # 配置文件
├── Dockerfile                  # 后端镜像
├── Dockerfile.frontend         # 前端镜像
├── docker-compose.yml          # 编排文件
└── nginx.conf                  # Nginx配置
```

## ⚙️ 配置

编辑 `config.yaml`：

```yaml
# 姿态估计
pose:
  mode: "balanced"         # lightweight / balanced / performance
  device: "cpu"            # cpu / cuda（有GPU用cuda）

# 球体追踪
ball:
  confidence_threshold: 0.3

# 事件检测阈值
events:
  angle_ranges:
    left_elbow: [90, 170]
    right_elbow: [90, 170]
    left_knee: [100, 170]
    right_knee: [100, 170]
  highlight_thresholds:
    wrist_speed_max: 1500
    body_lean_max: 35

# 慢放设置
slowmo:
  factor: 0.25             # 0.25 = 4倍慢放

# LLM配置（支持任意OpenAI兼容API）
llm:
  base_url: "https://api.openai.com/v1"
  api_key: "sk-..."
  model: "gpt-4.1-mini"
```

## 📊 分析输出

| 输出文件 | 说明 |
|---------|------|
| `full_analysis.mp4` | 原速视频 + 骨骼/球标注 + 指标HUD |
| `problems_slowmo.mp4` | 问题片段 4倍慢放 + 说明 + 改进建议 |
| `highlights_slowmo.mp4` | 精彩瞬间 4倍慢放 |
| `report.json` | 结构化运动参数 + 事件列表 |
| `coaching.md` | LLM教练建议全文 |

## 🔌 API 端点

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
# 运行全部测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_param_extractor.py -v
```

## 🤝 贡献

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📄 License

MIT License - 详见 [LICENSE](LICENSE)

## 🙏 致谢

- [MMPose](https://github.com/open-mmlab/mmpose) - RTMPose姿态估计
- [rtmlib](https://github.com/Tau-J/rtlib) - 轻量RTMPose推理
- [TrackNet](https://github.com/yastrebksv/TrackNet) - 球体追踪
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO系列
