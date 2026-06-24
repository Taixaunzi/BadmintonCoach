"""分析API — 触发分析、状态查询、结果获取"""
import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import load_config
from ..engine.pipeline import AnalysisPipeline
from ..llm.client import CoachLLM
from ..models.enums import AnalysisStatus
from ..models.schemas import AnalysisProgress

router = APIRouter()
config = load_config()

# 内存中的分析状态
_analysis_tasks: dict[str, dict] = {}


@router.post("/analysis/{video_id}")
async def start_analysis(video_id: str):
    """触发视频分析"""
    upload_dir = Path(config.app.upload_dir) / video_id
    if not upload_dir.exists():
        raise HTTPException(404, "视频不存在，请先上传")

    # 查找输入文件
    input_files = list(upload_dir.glob("input.*"))
    if not input_files:
        raise HTTPException(404, "未找到输入视频文件")
    input_path = str(input_files[0])

    output_dir = str(Path(config.app.output_dir) / video_id)

    # 初始化状态
    _analysis_tasks[video_id] = {
        "status": AnalysisStatus.PENDING,
        "progress": 0.0,
        "message": "等待开始...",
        "result": None,
        "coaching": None,
        "error": None,
    }

    # 后台运行分析
    asyncio.create_task(_run_analysis(video_id, input_path, output_dir))

    return {"video_id": video_id, "status": "started"}


async def _run_analysis(video_id: str, input_path: str, output_dir: str):
    """后台执行分析"""
    task = _analysis_tasks[video_id]
    try:
        def on_progress(prog: AnalysisProgress):
            task["status"] = prog.status
            task["progress"] = prog.progress
            task["message"] = prog.message

        # 在线程池中运行（CPU密集型）
        pipeline = AnalysisPipeline(config)
        result = await asyncio.to_thread(
            pipeline.run, input_path, output_dir, video_id, on_progress
        )
        task["result"] = result.model_dump()
        task["status"] = AnalysisStatus.DONE
        task["progress"] = 1.0

        # LLM教练反馈
        if config.llm.api_key:
            task["status"] = AnalysisStatus.LLM
            task["message"] = "AI教练分析中..."
            try:
                llm = CoachLLM(config.llm)
                report_path = result.output_files.get("report", "")
                if report_path and Path(report_path).exists():
                    with open(report_path) as f:
                        report = json.load(f)
                    coaching = await llm.get_coaching(report)
                    task["coaching"] = coaching
                    # 保存到文件
                    coaching_path = Path(output_dir) / "coaching.md"
                    coaching_path.write_text(coaching, encoding="utf-8")
            except Exception as e:
                task["coaching"] = f"LLM分析失败: {e}"

        task["status"] = AnalysisStatus.DONE
        task["message"] = "分析完成!"

    except Exception as e:
        task["status"] = AnalysisStatus.ERROR
        task["error"] = str(e)
        task["message"] = f"分析失败: {e}"


@router.get("/analysis/{video_id}/status")
async def get_status(video_id: str):
    """查询分析状态"""
    task = _analysis_tasks.get(video_id)
    if not task:
        raise HTTPException(404, "未找到分析任务")
    return {
        "video_id": video_id,
        "status": task["status"].value if isinstance(task["status"], AnalysisStatus) else task["status"],
        "progress": task["progress"],
        "message": task["message"],
    }


@router.get("/analysis/{video_id}/result")
async def get_result(video_id: str):
    """获取分析结果"""
    task = _analysis_tasks.get(video_id)
    if not task:
        raise HTTPException(404, "未找到分析任务")
    if task["status"] != AnalysisStatus.DONE:
        raise HTTPException(202, "分析尚未完成")
    return {
        "video_id": video_id,
        "result": task["result"],
        "coaching": task["coaching"],
    }


@router.get("/analysis/{video_id}/events")
async def get_events(video_id: str):
    """获取事件列表"""
    task = _analysis_tasks.get(video_id)
    if not task:
        raise HTTPException(404, "未找到分析任务")
    if not task["result"]:
        raise HTTPException(202, "分析尚未完成")
    return {"events": task["result"].get("events", [])}


@router.post("/llm/chat")
async def llm_chat(body: dict):
    """LLM自由对话（追问教练）"""
    message = body.get("message", "")
    video_id = body.get("video_id", "")
    context = None

    if video_id:
        task = _analysis_tasks.get(video_id)
        if task and task["result"]:
            context = task["result"]

    if not config.llm.api_key:
        raise HTTPException(400, "未配置LLM API Key")

    llm = CoachLLM(config.llm)
    reply = await llm.chat(message, context)
    return {"reply": reply}


@router.get("/config/llm")
async def get_llm_config():
    """获取LLM配置（隐藏API Key）"""
    return {
        "base_url": config.llm.base_url,
        "model": config.llm.model,
        "has_api_key": bool(config.llm.api_key),
    }
