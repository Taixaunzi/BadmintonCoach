"""WebSocket进度推送"""
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

# 连接池：video_id -> [websocket, ...]
_connections: dict[str, list[WebSocket]] = {}


@router.websocket("/ws/analysis/{video_id}")
async def analysis_progress_ws(websocket: WebSocket, video_id: str):
    """WebSocket实时推送分析进度"""
    await websocket.accept()

    if video_id not in _connections:
        _connections[video_id] = []
    _connections[video_id].append(websocket)

    try:
        while True:
            # 保持连接，等待客户端消息或断开
            data = await websocket.receive_text()
            # 客户端可发送ping
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        _connections[video_id].remove(websocket)
        if not _connections[video_id]:
            del _connections[video_id]


async def broadcast_progress(video_id: str, data: dict):
    """向指定video_id的所有WebSocket连接广播进度"""
    conns = _connections.get(video_id, [])
    message = json.dumps(data, ensure_ascii=False)
    for ws in conns:
        try:
            await ws.send_text(message)
        except Exception:
            pass
