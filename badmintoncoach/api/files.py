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
    # 安全：验证路径不逃逸出output_dir
    base = Path(config.app.output_dir).resolve()
    file_path = (base / video_id / filename).resolve()
    if not file_path.is_relative_to(base):
        raise HTTPException(403, "禁止访问")
    if not file_path.exists():
        raise HTTPException(404, "文件不存在")
    return FileResponse(file_path)
