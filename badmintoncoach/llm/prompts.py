"""LLM Prompt模板 — Skill Router模式"""

SYSTEM_PROMPT = """你是一位专业的羽毛球运动生物力学教练。
基于以下分析数据，给出专业、具体、可执行的教练反馈。

反馈要求：
1. 具体（引用精确测量值）
2. 可执行（告诉运动员改什么、怎么改）
3. 鼓励性（先肯定做得好的）
4. 优先级排序（影响最大的先说）
5. 用中文回答

输出JSON格式：
{
  "summary": "一句话总结",
  "strengths": ["优点1", "优点2"],
  "problems": [
    {"issue": "问题描述", "evidence": "数据支撑", "fix": "改进建议"}
  ],
  "highlights": ["精彩瞬间描述"],
  "training_tips": ["训练建议1", "训练建议2"],
  "risk_alerts": ["风险提示"],
  "overall_score": 75
}"""


def build_coaching_prompt(analysis_summary: dict) -> str:
    """构建教练反馈Prompt"""
    parts = ["## 分析数据\n"]

    # 基础信息
    parts.append(f"- 总帧数: {analysis_summary.get('total_frames', 'N/A')}")
    parts.append(f"- 时长: {analysis_summary.get('duration', 0):.1f}秒")
    parts.append(f"- FPS: {analysis_summary.get('fps', 0):.1f}")

    # 运动参数汇总
    summary = analysis_summary.get("frames_summary", {})
    parts.append(f"\n## 运动参数汇总")
    parts.append(f"- 平均手腕速度: {summary.get('avg_wrist_speed', 0):.0f} px/f")
    parts.append(f"- 平均身体倾斜: {summary.get('avg_body_lean', 0):.0f}°")

    # 事件
    events = analysis_summary.get("events", [])
    problems = [e for e in events if e.get("event_type") == "problem"]
    highlights = [e for e in events if e.get("event_type") == "highlight"]

    if problems:
        parts.append(f"\n## 检测到的问题 ({len(problems)}个)")
        for p in problems[:10]:  # 最多10个
            parts.append(
                f"- [{p.get('severity', 'warning')}] {p.get('description', '')} "
                f"(时间: {p.get('timestamp', 0):.1f}s) → {p.get('improvement', '')}"
            )

    if highlights:
        parts.append(f"\n## 精彩瞬间 ({len(highlights)}个)")
        for h in highlights[:5]:
            parts.append(
                f"- {h.get('description', '')} (时间: {h.get('timestamp', 0):.1f}s)"
            )

    parts.append("\n请基于以上数据给出教练反馈。")
    return "\n".join(parts)
