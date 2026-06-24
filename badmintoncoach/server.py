"""FastAPI 服务器入口"""
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.files import router as files_router
from .api.upload import router as upload_router
from .config import load_config

config = load_config()

app = FastAPI(title="BadmintonCoach", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 确保目录存在
Path(config.app.upload_dir).mkdir(parents=True, exist_ok=True)
Path(config.app.output_dir).mkdir(parents=True, exist_ok=True)

# 静态文件（输出视频）
app.mount("/output", StaticFiles(directory=config.app.output_dir), name="output")

app.include_router(upload_router, prefix="/api")
app.include_router(files_router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


if __name__ == "__main__":
    uvicorn.run(
        "badmintoncoach.server:app",
        host=config.app.host,
        port=config.app.port,
        reload=True,
    )
