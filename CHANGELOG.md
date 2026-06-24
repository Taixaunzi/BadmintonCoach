# Changelog

## [0.2.0] - 2026-06-24

### Added
- **TrackNet球追踪**: 深度学习模型追踪羽毛球，支持PyTorch/ONNX双后端
- **GPU加速**: 支持CUDA/MPS/ONNX/TensorRT，可配置设备选择
- **球拍检测**: 手腕区域形状+亮度分析，权重50%
- **球场检测**: HSV颜色分割识别绿色/蓝色/红色球场
- **运动员过滤**: 四维评分系统（球拍+身高+球场+位置）
- **事件检测v2**: 滑窗聚合+活动状态机+挥拍检测+多条件联合
- **中文HUD**: PIL渲染中文文字，替代OpenCV putText
- **球轨迹可视化**: 渐变色轨迹线+球速标签(km/h)
- **GPU检测工具**: `python -m badmintoncoach.tools.gpu_utils`
- **ONNX导出**: TrackNet模型导出为ONNX格式

### Fixed
- 路径穿越漏洞: files.py添加is_relative_to检查
- fps=0除零错误: 默认30fps兜底
- WebSocket进度未接入: 接入broadcast_progress广播
- LLM api_key空值: 提前抛出明确错误
- 多人关键点索引混乱: 改为单人模式
- 慢镜头精简: 去掉标题卡，只保留时间戳

### Changed
- 事件检测从单帧阈值改为滑窗聚合+状态机
- 运动员过滤从简单位置改为四维评分系统
- 球追踪从颜色检测改为TrackNet深度学习
- 视频要求: 需要近距离拍摄（运动员>25%画面）

## [0.1.0] - 2026-06-24

### Added
- 项目骨架: FastAPI + Vue 3 + TypeScript
- 关节角度计算: 8个关节，12个单元测试
- 骨骼姿态估计: RTMPose封装
- 视频标注: 骨骼+球+指标HUD
- 分析管道: 完整pipeline编排
- LLM教练: OpenAI兼容客户端+Skill Router Prompt
- 分析API: 触发/状态/结果/事件/对话
- WebSocket进度推送
- Vue 3前端: 上传/分析结果/设置页面
- Docker部署: docker-compose一键启动
- 设计文档+README+MIT License
