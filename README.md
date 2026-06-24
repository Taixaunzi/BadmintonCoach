# 🏸 BadmintonCoach

开源羽毛球视频AI分析系统 — 骨骼标定 + 球体追踪 + 运动参数 + LLM教练反馈

## ✨ 功能

- **骨骼姿态估计**：RTMPose 实时标注人体17个关键点
- **球体追踪**：TrackNet 热力图追踪羽毛球（亚像素精度）
- **运动参数提取**：关节角度、手腕速度、身体倾斜、轨迹分析
- **问题自动检测**：关节角度超标、动作偏差自动标记
- **精彩瞬间捕捉**：高速挥拍、极限救球自动识别
- **慢镜头回放**：问题/精彩片段 4倍慢放 + 文字叠加说明
- **LLM教练反馈**：OpenAI兼容API，支持任意LLM提供教练建议
- **Web交互界面**：Vue 3 前端，视频上传 → 分析 → 结果浏览

## 🏗️ 技术栈

| 组件 | 方案 |
|------|------|
| 后端 | FastAPI + Python 3.10+ |
| 前端 | Vue 3 + Vite + TypeScript |
| 骨骼 | RTMPose (rtmlib, ONNX) |
| 球追踪 | TrackNet (热力图, 3帧输入) |
| 视频 | OpenCV + FFmpeg |
| LLM | OpenAI兼容API (用户自配) |
| 部署 | Docker Compose |

## 🚀 快速开始

```bash
# 克隆
git clone https://github.com/Taixaunzi/BadmintonCoach.git
cd BadmintonCoach

# 后端
pip install -r requirements.txt
python -m badmintoncoach.server

# 前端
cd frontend
npm install
npm run dev

# 或 Docker 一键启动
docker-compose up
```

## 📁 项目结构

```
BadmintonCoach/
├── badmintoncoach/         # Python包
│   ├── server.py           # FastAPI入口
│   ├── config.py           # 配置管理
│   ├── api/                # REST API路由
│   ├── engine/             # 分析引擎（姿态/球追踪/参数/事件/标注/慢放）
│   ├── llm/                # LLM客户端和Prompt
│   └── models/             # 数据模型
├── frontend/               # Vue 3 前端
├── models/                 # 预训练模型（ONNX）
├── docs/                   # 设计文档
├── tests/                  # 测试
├── config.yaml             # 配置文件
├── docker-compose.yml
└── requirements.txt
```

## 📊 分析输出

| 输出 | 说明 |
|------|------|
| `full_analysis.mp4` | 原速视频 + 骨骼/球标注 + 指标HUD |
| `problems_slowmo.mp4` | 问题片段 4倍慢放 + 说明 + 改进建议 |
| `highlights_slowmo.mp4` | 精彩瞬间 4倍慢放 |
| `report.json` | 结构化运动参数 |
| `coaching.md` | LLM教练建议 |

## ⚙️ 配置

```yaml
# config.yaml
pose:
  mode: "balanced"         # lightweight / balanced / performance
  device: "cpu"            # cpu / cuda

ball:
  confidence_threshold: 0.3

llm:
  base_url: "https://api.openai.com/v1"
  api_key: "sk-..."
  model: "gpt-4.1-mini"
```

## 📄 License

MIT
