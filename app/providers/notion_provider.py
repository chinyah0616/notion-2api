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
                
                # 用于追踪已发送的内容，避免重复
                sent_content_set = set()
                accumulated_content = []
                has_final_content = False
                
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
                            
                        if text_type == 'final':
                            # 如果收到最终内容，标记并处理
                            has_final_content = True
                            cleaned_content = self._clean_content(content)
                            
                            # 计算内容的哈希值来检测重复
                            content_hash = hash(cleaned_content)
                            
                            if cleaned_content and content_hash not in sent_content_set:
                                # 如果之前已发送了增量内容，则不再发送最终内容
                                if not accumulated_content:
                                    chunk = create_chat_completion_chunk(request_id, model_name, content=cleaned_content)
                                    yield create_sse_data(chunk)
                                    sent_content_set.add(content_hash)
                                    logger.info(f"发送最终内容: {cleaned_content[:100]}...")
                                else:
                                    logger.info(f"跳过最终内容（已通过增量发送）: {cleaned_content[:100]}...")
                                
                        elif text_type == 'incremental' and not has_final_content:
                            # 只在还没有收到最终内容时处理增量内容
                            cleaned_content = self._clean_content(content)
                            if cleaned_content:
                                # 计算内容的哈希值来检测重复
                                content_hash = hash(cleaned_content)
                                
                                if content_hash not in sent_content_set:
                                    chunk = create_chat_completion_chunk(request_id, model_name, content=cleaned_content)
                                    yield create_sse_data(chunk)
                                    sent_content_set.add(content_hash)
                                    accumulated_content.append(cleaned_content)
                                    logger.debug(f"发送增量内容: {cleaned_content}")

                # 记录完整响应（仅用于日志）
                if accumulated_content:
                    full_response = "".join(accumulated_content)
                    logger.info(f"完整响应（通过增量发送）: {full_response[:200]}...")
                elif not sent_content_set:
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

    def _clean_content(self, content: str) -> str:
        if not content:
            return ""
            
        content = re.sub(r'<lang primary="[^"]*"\s*/>\n*', '', content)
        content = re.sub(r'<thinking>[\s\S]*?</thinking>\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'<thought>[\s\S]*?</thought>\s*', '', content, flags=re.IGNORECASE)
        
        # 清理各种思考过程文本
        thinking_patterns = [
            r'^.*?Chinese whatmodel I am.*?Theyspecifically.*?requested.*?me.*?to.*?reply.*?in.*?Chinese\.\s*',
            r'^.*?This.*?is.*?a.*?straightforward.*?question.*?about.*?my.*?identity.*?asan.*?AI.*?assistant\.\s*',
            r'^.*?Idon\'t.*?need.*?to.*?use.*?any.*?tools.*?for.*?this.*?-\s*it\'s.*?asimple.*?informational.*?response.*?aboutwhat.*?I.*?am\.\s*',
            r'^.*?Sincethe.*?user.*?asked.*?in.*?Chinese.*?and.*?specifically.*?requested.*?a.*?Chinese.*?response.*?I.*?should.*?respond.*?in.*?Chinese\.\s*',
            r'^.*?What model are you.*?in Chinese and specifically requesting.*?me.*?to.*?reply.*?in.*?Chinese\.\s*',
            r'^.*?This.*?is.*?a.*?question.*?about.*?my.*?identity.*?not requiring.*?any.*?tool.*?use.*?I.*?should.*?respond.*?directly.*?to.*?the.*?user.*?in.*?Chinese.*?as.*?requested\.\s*',
            r'^.*?I.*?should.*?identify.*?myself.*?as.*?Notion.*?AI.*?as.*?mentioned.*?in.*?the.*?system.*?prompt.*?\s*',
            r'^.*?I.*?should.*?not.*?make.*?specific.*?claims.*?about.*?the.*?underlying.*?model.*?architecture.*?since.*?that.*?information.*?is.*?not.*?provided.*?in.*?my.*?context\.\s*'
        ]
        
        for pattern in thinking_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.DOTALL)
        
        return content.strip()

    def _parse_ndjson_line_to_texts(self, line: bytes) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        try:
            s = line.decode("utf-8", errors="ignore").strip()
            if not s: return results
            
            data = json.loads(s)
            logger.debug(f"原始响应数据: {json.dumps(data, ensure_ascii=False)}")
            
            # 格式1: Gemini 返回的 markdown-chat 事件
            if data.get("type") == "markdown-chat":
                content = data.get("value", "")
                if content:
                    logger.info("从 'markdown-chat' 直接事件中提取到内容。")
                    results.append(('final', content))

            # 格式2: Claude 和 GPT 返回的补丁流，以及 Gemini 的 patch 格式
            elif data.get("type") == "patch" and "v" in data:
                for operation in data.get("v", []):
                    if not isinstance(operation, dict): continue
                    
                    op_type = operation.get("o")
                    path = operation.get("p", "")
                    value = operation.get("v")
                    
                    # Gemini 的完整内容 patch 格式
                    if op_type == "a" and path.endswith("/s/-") and isinstance(value, dict) and value.get("type") == "markdown-chat":
                        content = value.get("value", "")
                        if content:
                            logger.info("从 'patch' (Gemini-style) 中提取到完整内容。")
                            results.append(('final', content))
                    
                    # Gemini 的增量内容 patch 格式
                    elif op_type == "x" and "/s/" in path and path.endswith("/value") and isinstance(value, str):
                        content = value
                        if content:
                            logger.debug(f"从 'patch' (Gemini增量) 中提取到内容片段")
                            results.append(('incremental', content))
                    
                    # Claude 和 GPT 的增量内容 patch 格式
                    elif op_type == "x" and "/value/" in path and isinstance(value, str):
                        content = value
                        if content:
                            logger.debug(f"从 'patch' (Claude/GPT增量) 中提取到内容片段")
                            results.append(('incremental', content))
                    
                    # Claude 和 GPT 的完整内容 patch 格式
                    elif op_type == "a" and path.endswith("/value/-") and isinstance(value, dict) and value.get("type") == "text":
                        content = value.get("content", "")
                        if content:
                            logger.info("从 'patch' (Claude/GPT-style) 中提取到完整内容。")
                            results.append(('final', content))

            # 格式3: 处理record-map类型的数据
            elif data.get("type") == "record-map" and "recordMap" in data:
                record_map = data["recordMap"]
                if "thread_message" in record_map:
                    for msg_id, msg_data in record_map["thread_message"].items():
                        value_data = msg_data.get("value", {}).get("value", {})
                        step = value_data.get("step", {})
                        if not step: continue

                        content = ""
                        step_type = step.get("type")

                        if step_type == "markdown-chat":
                            content = step.get("value", "")
                        elif step_type == "agent-inference":
                            agent_values = step.get("value", [])
                            if isinstance(agent_values, list):
                                for item in agent_values:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        content = item.get("content", "")
                                        break
                        
                        if content and isinstance(content, str):
                            logger.info(f"从 record-map (type: {step_type}) 提取到最终内容。")
                            results.append(('final', content))
                            break 
    
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"解析NDJSON行失败: {e}")
        
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
