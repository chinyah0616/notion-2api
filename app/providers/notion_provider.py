# app/providers/notion_provider.py
import json
import time
import logging
import uuid
import re
import cloudscraper
from typing import Dict, Any, AsyncGenerator, List, Optional, Tuple
from datetime import datetime

from fastapi import HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.providers.base_provider import BaseProvider
from app.utils.sse_utils import create_sse_data, create_chat_completion_chunk, DONE_CHUNK

# 设置日志记录器
logger = logging.getLogger(__name__)

class NotionAIProvider(BaseProvider):
    def __init__(self):
        self.scraper = cloudscraper.create_scraper()
        self.api_endpoints = {
            "runInference": "https://www.notion.so/api/v3/runInferenceTranscript",
            "saveTransactions": "https://www.notion.so/api/v3/saveTransactionsFanout"
        }
        
        if not all([settings.NOTION_COOKIE, settings.NOTION_SPACE_ID, settings.NOTION_USER_ID]):
            raise ValueError("配置错误: NOTION_COOKIE, NOTION_SPACE_ID 和 NOTION_USER_ID 必须在 .env 文件中全部设置。")

        self._warmup_session()

    def _warmup_session(self):
        try:
            logger.info("正在进行会话预热 (Session Warm-up)...")
            headers = self._prepare_headers()
            headers.pop("Accept", None)
            response = self.scraper.get("https://www.notion.so/", headers=headers, timeout=30)
            response.raise_for_status()
            logger.info("会话预热成功。")
        except Exception as e:
            logger.error(f"会话预热失败: {e}", exc_info=True)
            
    async def _create_thread(self, thread_type: str) -> str:
        thread_id = str(uuid.uuid4())
        payload = {
            "requestId": str(uuid.uuid4()),
            "transactions": [{
                "id": str(uuid.uuid4()),
                "spaceId": settings.NOTION_SPACE_ID,
                "operations": [{
                    "pointer": {"table": "thread", "id": thread_id, "spaceId": settings.NOTION_SPACE_ID},
                    "path": [],
                    "command": "set",
                    "args": {
                        "id": thread_id, "version": 1, "parent_id": settings.NOTION_SPACE_ID,
                        "parent_table": "space", "space_id": settings.NOTION_SPACE_ID,
                        "created_time": int(time.time() * 1000),
                        "created_by_id": settings.NOTION_USER_ID, "created_by_table": "notion_user",
                        "messages": [], "data": {}, "alive": True, "type": thread_type
                    }
                }]
            }]
        }
        try:
            logger.info(f"正在创建新的对话线程 (type: {thread_type})...")
            response = await run_in_threadpool(
                lambda: self.scraper.post(
                    self.api_endpoints["saveTransactions"],
                    headers=self._prepare_headers(),
                    json=payload,
                    timeout=20
                )
            )
            response.raise_for_status()
            logger.info(f"对话线程创建成功, Thread ID: {thread_id}")
            return thread_id
        except Exception as e:
            logger.error(f"创建对话线程失败: {e}", exc_info=True)
            raise Exception("无法创建新的对话线程。")

    async def chat_completion(self, request_data: Dict[str, Any]):
        stream = request_data.get("stream", True)

        async def stream_generator() -> AsyncGenerator[bytes, None]:
            request_id = f"chatcmpl-{uuid.uuid4()}"
            
            try:
                model_name = request_data.get("model", settings.DEFAULT_MODEL)
                mapped_model = settings.MODEL_MAP.get(model_name, "anthropic-sonnet-alt")
                
                thread_type = "markdown-chat" if mapped_model.startswith("vertex-") else "workflow"
                
                thread_id = await self._create_thread(thread_type)
                payload = self._prepare_payload(request_data, thread_id, mapped_model, thread_type)
                headers = self._prepare_headers()

                # 发送角色信息
                role_chunk = create_chat_completion_chunk(request_id, model_name, role="assistant")
                yield create_sse_data(role_chunk)

                def sync_stream_iterator():
                    try:
                        logger.info(f"请求 Notion AI URL: {self.api_endpoints['runInference']}")
                        logger.info(f"请求体: {json.dumps(payload, indent=2, ensure_ascii=False)}")
                        
                        response = self.scraper.post(
                            self.api_endpoints['runInference'], headers=headers, json=payload, stream=True,
                            timeout=settings.API_REQUEST_TIMEOUT
                        )
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if line:
                                yield line
                    except Exception as e:
                        yield e

                sync_gen = sync_stream_iterator()
                
                # 用于累积完整内容（仅用于日志）
                accumulated_content = []
                event_count = 0
                
                while True:
                    line = await run_in_threadpool(lambda: next(sync_gen, None))
                    if line is None:
                        break
                    if isinstance(line, Exception):
                        raise line

                    parsed_results = self._parse_ndjson_line_to_texts(line)
                    
                    for text_type, content in parsed_results:
                        if not content:
                            continue
                        
                        event_count += 1
                        logger.debug(f"Event #{event_count} - Type: {text_type}, Content: {repr(content[:100]) if len(content) > 100 else repr(content)}")
                        
                        # 处理所有类型的内容（incremental, update, append等）
                        if text_type in ['incremental', 'update', 'append']:
                            # 直接发送原始内容，保留所有格式
                            chunk = create_chat_completion_chunk(request_id, model_name, content=content)
                            yield create_sse_data(chunk)
                            accumulated_content.append(content)

                # 记录完整响应（仅用于日志）
                full_response = "".join(accumulated_content)
                if full_response:
                    logger.info(f"完整响应（{event_count}个事件）: {full_response[:200]}...")
                else:
                    logger.warning("警告: Notion 返回的数据流中未提取到任何有效文本。")

                # 发送结束标记
                final_chunk = create_chat_completion_chunk(request_id, model_name, finish_reason="stop")
                yield create_sse_data(final_chunk)
                yield DONE_CHUNK

            except Exception as e:
                error_message = f"处理 Notion AI 流时发生意外错误: {str(e)}"
                logger.error(error_message, exc_info=True)
                error_chunk = {"error": {"message": error_message, "type": "internal_server_error"}}
                yield create_sse_data(error_chunk)
                yield DONE_CHUNK

        if stream:
            return StreamingResponse(
                stream_generator(), 
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
                    "Transfer-Encoding": "chunked",
                }
            )
        else:
            raise HTTPException(status_code=400, detail="此端点当前仅支持流式响应 (stream=true)。")

    def _prepare_headers(self) -> Dict[str, str]:
        cookie_source = (settings.NOTION_COOKIE or "").strip()
        cookie_header = cookie_source if "=" in cookie_source else f"token_v2={cookie_source}"

        return {
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
            "Cookie": cookie_header,
            "x-notion-space-id": settings.NOTION_SPACE_ID,
            "x-notion-active-user-header": settings.NOTION_USER_ID,
            "x-notion-client-version": settings.NOTION_CLIENT_VERSION,
            "notion-audit-log-platform": "web",
            "Origin": "https://www.notion.so",
            "Referer": "https://www.notion.so/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        }

    def _normalize_block_id(self, block_id: str) -> str:
        if not block_id: return block_id
        b = block_id.replace("-", "").strip()
        if len(b) == 32 and re.fullmatch(r"[0-9a-fA-F]{32}", b):
            return f"{b[0:8]}-{b[8:12]}-{b[12:16]}-{b[16:20]}-{b[20:]}"
        return block_id

    def _prepare_payload(self, request_data: Dict[str, Any], thread_id: str, mapped_model: str, thread_type: str) -> Dict[str, Any]:
        req_block_id = request_data.get("notion_block_id") or settings.NOTION_BLOCK_ID
        normalized_block_id = self._normalize_block_id(req_block_id) if req_block_id else None

        context_value: Dict[str, Any] = {
            "timezone": "Asia/Shanghai",
            "spaceId": settings.NOTION_SPACE_ID,
            "userId": settings.NOTION_USER_ID,
            "userEmail": settings.NOTION_USER_EMAIL,
            "currentDatetime": datetime.now().astimezone().isoformat(),
        }
        if normalized_block_id:
            context_value["blockId"] = normalized_block_id

        config_value: Dict[str, Any]
        
        if mapped_model.startswith("vertex-"):
            logger.info(f"检测到 Gemini 模型 ({mapped_model})，应用特定的 config 和 context。")
            context_value.update({
                "userName": f" {settings.NOTION_USER_NAME}",
                "spaceName": f"{settings.NOTION_USER_NAME}的 Notion",
                "spaceViewId": "2008eefa-d0dc-80d5-9e67-000623befd8f",
                "surface": "ai_module"
            })
            config_value = {
                "type": thread_type,
                "model": mapped_model,
                "useWebSearch": True,
                "enableAgentAutomations": False, "enableAgentIntegrations": False,
                "enableBackgroundAgents": False, "enableCodegenIntegration": False,
                "enableCustomAgents": False, "enableExperimentalIntegrations": False,
                "enableLinkedDatabases": False, "enableAgentViewVersionHistoryTool": False,
                "searchScopes": [{"type": "everything"}], "enableDatabaseAgents": False,
                "enableAgentComments": False, "enableAgentForms": False,
                "enableAgentMakesFormulas": False, "enableUserSessionContext": False,
                "modelFromUser": True, "isCustomAgent": False
            }
        else:
            context_value.update({
                "userName": settings.NOTION_USER_NAME,
                "surface": "workflows"
            })
            config_value = {
                "type": thread_type,
                "model": mapped_model,
                "useWebSearch": True,
            }

        transcript = [
            {"id": str(uuid.uuid4()), "type": "config", "value": config_value},
            {"id": str(uuid.uuid4()), "type": "context", "value": context_value}
        ]
      
        for msg in request_data.get("messages", []):
            if msg.get("role") == "user":
                transcript.append({
                    "id": str(uuid.uuid4()),
                    "type": "user",
                    "value": [[msg.get("content")]],
                    "userId": settings.NOTION_USER_ID,
                    "createdAt": datetime.now().astimezone().isoformat()
                })
            elif msg.get("role") == "assistant":
                transcript.append({"id": str(uuid.uuid4()), "type": "agent-inference", "value": [{"type": "text", "content": msg.get("content")}]})

        payload = {
            "traceId": str(uuid.uuid4()),
            "spaceId": settings.NOTION_SPACE_ID,
            "transcript": transcript,
            "threadId": thread_id,
            "createThread": False,
            "isPartialTranscript": True,
            "asPatchResponse": True,
            "generateTitle": True,
            "saveAllThreadOperations": True,
            "threadType": thread_type
        }

        if mapped_model.startswith("vertex-"):
            logger.info("为 Gemini 请求添加 debugOverrides。")
            payload["debugOverrides"] = {
                "emitAgentSearchExtractedResults": True,
                "cachedInferences": {},
                "annotationInferences": {},
                "emitInferences": False
            }
        
        return payload

    def _parse_ndjson_line_to_texts(self, line: bytes) -> List[Tuple[str, str]]:
        """
        解析NDJSON行，提取所有类型的文本内容
        返回: [(类型, 内容), ...]
        类型可以是: 'update', 'append', 'incremental' 等
        """
        results: List[Tuple[str, str]] = []
        try:
            s = line.decode("utf-8", errors="ignore").strip()
            if not s: return results
            
            data = json.loads(s)
            
            # 处理不同类型的事件
            event_type = data.get("type", "")
            
            # 1. 处理 patch 类型（最常见）
            if event_type == "patch" and "v" in data:
                for operation in data.get("v", []):
                    if not isinstance(operation, dict): continue
                    
                    op_type = operation.get("o")
                    path = operation.get("p", "")
                    value = operation.get("v")
                    
                    # 处理不同的 patch 操作
                    if op_type == "x" and isinstance(value, str) and value:
                        # 判断是 update 还是 append
                        if "update_task_message" in path or "/task_message/" in path:
                            logger.debug(f"检测到 update_task_message: {repr(value[:50])}")
                            results.append(('update', value))
                        elif "append_to_message_content" in path or "/message_content/" in path:
                            logger.debug(f"检测到 append_to_message_content: {repr(value[:50])}")
                            results.append(('append', value))
                        elif "/s/" in path and path.endswith("/value"):
                            # Gemini 的增量内容格式
                            logger.debug(f"检测到 Gemini 增量内容: {repr(value[:50])}")
                            results.append(('incremental', value))
                        elif "/value/" in path:
                            # Claude/GPT 的增量内容格式
                            logger.debug(f"检测到 Claude/GPT 增量内容: {repr(value[:50])}")
                            results.append(('incremental', value))
                        else:
                            # 其他未知格式，也尝试处理
                            logger.debug(f"检测到未知格式内容 (path: {path}): {repr(value[:50])}")
                            results.append(('incremental', value))
                    
                    # 处理设置操作（可能包含初始内容）
                    elif op_type == "s" and isinstance(value, str) and value:
                        logger.debug(f"检测到设置操作内容: {repr(value[:50])}")
                        results.append(('update', value))
            
            # 2. 处理直接的消息事件
            elif event_type == "update_task_message":
                content = data.get("value", "")
                if content:
                    logger.debug(f"检测到直接 update_task_message 事件: {repr(content[:50])}")
                    results.append(('update', content))
            
            elif event_type == "append_to_message_content":
                content = data.get("value", "")
                if content:
                    logger.debug(f"检测到直接 append_to_message_content 事件: {repr(content[:50])}")
                    results.append(('append', content))
            
            # 3. 处理 markdown-chat 类型（某些模型可能使用）
            elif event_type == "markdown-chat":
                content = data.get("value", "")
                if content:
                    logger.debug(f"检测到 markdown-chat 事件: {repr(content[:50])}")
                    results.append(('incremental', content))
            
            # 记录未处理的事件类型（用于调试）
            if not results and event_type:
                logger.debug(f"未处理的事件类型: {event_type}, 数据: {json.dumps(data, ensure_ascii=False)[:200]}")
    
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"解析NDJSON行失败: {e}, 原始数据: {line.decode('utf-8', errors='ignore')[:200]}")
        
        return results

    async def get_models(self) -> JSONResponse:
        model_data = {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": int(time.time()), "owned_by": "lzA6"}
                for name in settings.KNOWN_MODELS
            ]
        }
        return JSONResponse(content=model_data)
