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
