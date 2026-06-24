"""视频上传API"""
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..config import load_config

router = APIRouter()
config = load_config()

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """上传视频，返回video_id"""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的格式: {ext}，支持: {ALLOWED_EXTENSIONS}")

    video_id = str(uuid.uuid4())[:8]
    upload_dir = Path(config.app.upload_dir) / video_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / f"input{ext}"
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        shutil.rmtree(upload_dir)
        raise HTTPException(400, "文件过大（最大500MB）")

    with open(file_path, "wb") as f:
        f.write(content)

    return {
        "video_id": video_id,
        "filename": file.filename,
        "size": len(content),
        "path": str(file_path),
    }
