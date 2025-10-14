# app/utils/sse_utils.py
import json
import time
from typing import Dict, Any, Optional

DONE_CHUNK = b"data: [DONE]\n\n"

def create_sse_data(data: Dict[str, Any]) -> bytes:
    """将数据转换为 SSE 格式"""
    return f"data: {json.dumps(data)}\n\n".encode('utf-8')

def create_chat_completion_chunk(
    request_id: str,
    model: str,
    content: Optional[str] = None,
    finish_reason: Optional[str] = None,
    role: Optional[str] = None
) -> Dict[str, Any]:
    """创建聊天完成的数据块"""
    delta: Dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content

    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }
        ]
    }
