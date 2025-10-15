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
                
                # 用于跟踪内容
                accumulated_content = []
                event_count = 0
                sent_content_set = set()  # 避免重复发送相同内容
                
                while True:
                    line = await run_in_threadpool(lambda: next(sync_gen, None))
                    if line is None:
                        break
                    if isinstance(line, Exception):
                        raise line

                    event_count += 1
                    parsed_results = self._parse_ndjson_line_comprehensive(line, event_count)
                    
                    for content_type, content in parsed_results:
                        if not content:
                            continue
                        
                        # 检查是否已发送过这个内容片段
                        content_hash = hash(content)
                        if content_hash in sent_content_set:
                            logger.debug(f"跳过重复内容: {repr(content[:50])}")
                            continue
                        
                        sent_content_set.add(content_hash)
                        
                        # 发送内容
                        chunk = create_chat_completion_chunk(request_id, model_name, content=content)
                        yield create_sse_data(chunk)
                        accumulated_content.append(content)
                        
                        if event_count <= 3:
                            logger.info(f"事件#{event_count} - 类型:{content_type} - 内容: {repr(content[:100])}")
                        else:
                            logger.debug(f"事件#{event_count} - 类型:{content_type} - 内容长度: {len(content)}")

                # 记录完整响应
                full_response = "".join(accumulated_content)
                if full_response:
                    logger.info(f"完整响应（{event_count}个事件）: {full_response[:200]}...")
                    if len(full_response) < 50:
                        logger.warning(f"响应内容较短，可能有内容丢失: {repr(full_response)}")
                else:
                    logger.warning(f"警告: 处理了{event_count}个事件但未提取到任何有效文本。")

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
                    "X-Accel-Buffering": "no",
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

    def _parse_ndjson_line_comprehensive(self, line: bytes, event_num: int) -> List[Tuple[str, str]]:
        """
        综合解析所有可能的数据格式，确保不丢失任何内容
        """
        results: List[Tuple[str, str]] = []
        try:
            s = line.decode("utf-8", errors="ignore").strip()
            if not s: return results
            
            data = json.loads(s)
            
            # 记录前几个事件的完整数据用于调试
            if event_num <= 3:
                logger.info(f"事件#{event_num} 原始数据类型: {data.get('type')}")
                logger.debug(f"事件#{event_num} 完整数据: {json.dumps(data, ensure_ascii=False)[:500]}")
            
            # 1. 处理直接的文本事件（如 markdown-chat）
            if data.get("type") == "markdown-chat":
                content = data.get("value", "")
                if content:
                    logger.info(f"[markdown-chat] 提取内容: {repr(content[:100])}")
                    results.append(('direct', content))
            
            # 2. 处理 update 类型事件
            elif data.get("type") == "update":
                # 尝试多种路径提取内容
                if "value" in data:
                    value = data["value"]
                    if isinstance(value, str):
                        if value:
                            logger.info(f"[update-string] 提取内容: {repr(value[:100])}")
                            results.append(('update', value))
                    elif isinstance(value, dict):
                        # 检查各种可能的内容路径
                        for key in ["content", "text", "message", "value"]:
                            if key in value:
                                content = value[key]
                                if isinstance(content, str) and content:
                                    logger.info(f"[update-dict-{key}] 提取内容: {repr(content[:100])}")
                                    results.append(('update', content))
                                    break
            
            # 3. 处理 patch 类型事件（增量更新）
            elif data.get("type") == "patch" and "v" in data:
                for operation in data.get("v", []):
                    if not isinstance(operation, dict): 
                        continue
                    
                    op_type = operation.get("o")
                    path = operation.get("p", "")
                    value = operation.get("v")
                    
                    # 3.1 处理替换操作（可能包含初始内容）
                    if op_type == "r" and isinstance(value, str) and value:
                        logger.info(f"[patch-replace] 路径:{path} 内容: {repr(value[:100])}")
                        results.append(('replace', value))
                    
                    # 3.2 处理设置操作（可能包含初始内容）
                    elif op_type == "s" and isinstance(value, str) and value:
                        logger.info(f"[patch-set] 路径:{path} 内容: {repr(value[:100])}")
                        results.append(('set', value))
                    
                    # 3.3 处理扩展操作（增量内容）
                    elif op_type == "x" and isinstance(value, str) and value:
                        # Gemini 格式
                        if "/s/" in path and path.endswith("/value"):
                            logger.debug(f"[patch-extend-gemini] 内容片段")
                            results.append(('extend', value))
                        # Claude/GPT 格式
                        elif "/value/" in path:
                            logger.debug(f"[patch-extend-claude] 内容片段")
                            results.append(('extend', value))
                        # 通用格式
                        else:
                            logger.debug(f"[patch-extend-generic] 路径:{path}")
                            results.append(('extend', value))
                    
                    # 3.4 处理添加操作（可能是完整内容）
                    elif op_type == "a":
                        if isinstance(value, str) and value:
                            logger.info(f"[patch-add-string] 内容: {repr(value[:100])}")
                            results.append(('add', value))
                        elif isinstance(value, dict):
                            # 处理嵌套的内容
                            if value.get("type") == "text" and "content" in value:
                                content = value["content"]
                                if content:
                                    logger.info(f"[patch-add-text] 内容: {repr(content[:100])}")
                                    results.append(('add', content))
                            elif value.get("type") == "markdown-chat" and "value" in value:
                                content = value["value"]
                                if content:
                                    logger.info(f"[patch-add-markdown] 内容: {repr(content[:100])}")
                                    results.append(('add', content))
            
            # 4. 处理 record-map 类型（可能包含完整响应）
            elif data.get("type") == "record-map" and "recordMap" in data:
                record_map = data["recordMap"]
                # 检查 thread_message
                if "thread_message" in record_map:
                    for msg_id, msg_data in record_map["thread_message"].items():
                        value_data = msg_data.get("value", {}).get("value", {})
                        step = value_data.get("step", {})
                        
                        if step:
                            content = self._extract_content_from_step(step)
                            if content:
                                logger.info(f"[record-map] 提取内容: {repr(content[:100])}")
                                results.append(('record', content))
                                break  # 通常只需要第一个
            
            # 5. 处理其他可能的格式
            elif "content" in data and isinstance(data["content"], str):
                content = data["content"]
                if content:
                    logger.info(f"[root-content] 提取内容: {repr(content[:100])}")
                    results.append(('root', content))
            elif "text" in data and isinstance(data["text"], str):
                content = data["text"]
                if content:
                    logger.info(f"[root-text] 提取内容: {repr(content[:100])}")
                    results.append(('root', content))
    
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"解析事件#{event_num}失败: {e}")
        
        return results

    def _extract_content_from_step(self, step: Dict[str, Any]) -> Optional[str]:
        """从step对象中提取内容"""
        step_type = step.get("type")
        
        if step_type == "markdown-chat":
            return step.get("value", "")
        elif step_type == "agent-inference":
            agent_values = step.get("value", [])
            if isinstance(agent_values, list):
                for item in agent_values:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("content", "")
        elif step_type == "text":
            return step.get("content", "")
        
        # 尝试直接获取value
        if "value" in step and isinstance(step["value"], str):
            return step["value"]
        
        return None

    async def get_models(self) -> JSONResponse:
        model_data = {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": int(time.time()), "owned_by": "lzA6"}
                for name in settings.KNOWN_MODELS
            ]
        }
        return JSONResponse(content=model_data)
