# app/providers/notion_provider.py
import json
import time
import logging
import uuid
import re
import cloudscraper
import threading
from itertools import cycle
from typing import Dict, Any, AsyncGenerator, List, Optional, Tuple
from datetime import datetime

from fastapi import HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings, NotionAccount
from app.providers.base_provider import BaseProvider
from app.utils.sse_utils import create_sse_data, create_chat_completion_chunk, DONE_CHUNK

logger = logging.getLogger(__name__)

class NotionAIProvider(BaseProvider):
    def __init__(self):
        self.scraper = cloudscraper.create_scraper()
        self.api_endpoints = {
            "runInference": "https://www.notion.so/api/v3/runInferenceTranscript",
            "saveTransactions": "https://www.notion.so/api/v3/saveTransactionsFanout"
        }
        
        if not settings.ACCOUNTS:
            raise ValueError("配置错误: 未找到任何 Notion 账号。")
        
        self.account_cycler = cycle(settings.ACCOUNTS)
        self.lock = threading.Lock()

        # 【核心修正】严格使用您提供的、不带结尾的正则表达式，不做任何改动
        self.lang_tag_pattern = re.compile(r'<lang primary="(?:zh(?:-C?N?)?)?')
        
        # 简化状态管理，只保留去重功能
        self.last_content = ""
        self.content_buffer = []

        self._warmup_session()
        
    def _get_next_account(self) -> NotionAccount:
        with self.lock:
            account = next(self.account_cycler)
            logger.info(f"使用账号: {account.user_name or account.user_id}")
            return account

    def _warmup_session(self):
        try:
            logger.info("正在进行会话预热 (Session Warm-up)...")
            account = self._get_next_account()
            headers = self._prepare_headers(account)
            headers.pop("Accept", None)
            response = self.scraper.get("https://www.notion.so/", headers=headers, timeout=30)
            response.raise_for_status()
            logger.info("会话预热成功。")
        except Exception as e:
            logger.error(f"会话预热失败: {e}", exc_info=True)
            
    async def _create_thread(self, thread_type: str, account: NotionAccount) -> str:
        thread_id = str(uuid.uuid4())
        payload = {
            "requestId": str(uuid.uuid4()),
            "transactions": [{
                "id": str(uuid.uuid4()),
                "spaceId": account.space_id,
                "operations": [{
                    "pointer": {"table": "thread", "id": thread_id, "spaceId": account.space_id},
                    "path": [],
                    "command": "set",
                    "args": {
                        "id": thread_id, "version": 1, "parent_id": account.space_id,
                        "parent_table": "space", "space_id": account.space_id,
                        "created_time": int(time.time() * 1000),
                        "created_by_id": account.user_id, "created_by_table": "notion_user",
                        "messages": [], "data": {}, "alive": True, "type": thread_type
                    }
                }]
            }]
        }
        try:
            logger.info(f"正在为账号 {account.user_id} 创建新的对话线程...")
            response = await run_in_threadpool(
                lambda: self.scraper.post(
                    self.api_endpoints["saveTransactions"],
                    headers=self._prepare_headers(account),
                    json=payload,
                    timeout=20
                )
            )
            response.raise_for_status()
            logger.info(f"对话线程创建成功, Thread ID: {thread_id}")
            return thread_id
        except Exception as e:
            logger.error(f"为账号 {account.user_id} 创建对话线程失败: {e}", exc_info=True)
            raise Exception("无法创建新的对话线程。")

    async def chat_completion(self, request_data: Dict[str, Any]):
        stream = request_data.get("stream", True)
        account = self._get_next_account()

        async def stream_generator() -> AsyncGenerator[bytes, None]:
            request_id = f"chatcmpl-{uuid.uuid4()}"
            
            try:
                model_name = request_data.get("model", settings.DEFAULT_MODEL)
                mapped_model = settings.MODEL_MAP.get(model_name, "anthropic-sonnet-alt")
                thread_type = "markdown-chat" if mapped_model.startswith("vertex-") else "workflow"
                thread_id = await self._create_thread(thread_type, account)
                payload = self._prepare_payload(request_data, thread_id, mapped_model, account)
                headers = self._prepare_headers(account)

                role_chunk = create_chat_completion_chunk(request_id, model_name, role="assistant")
                yield create_sse_data(role_chunk)

                # 重置去重状态
                self.last_content = ""
                self.content_buffer = []

                def sync_stream_iterator():
                    try:
                        logger.info(f"请求 Notion AI URL: {self.api_endpoints['runInference']}")
                        response = self.scraper.post(
                            self.api_endpoints['runInference'], headers=headers, json=payload, stream=True,
                            timeout=settings.API_REQUEST_TIMEOUT
                        )
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if line: yield line
                    except Exception as e:
                        yield e

                sync_gen = sync_stream_iterator()
                
                accumulated_content = []
                event_count = 0
                
                while True:
                    line = await run_in_threadpool(lambda: next(sync_gen, None))
                    if line is None: break
                    if isinstance(line, Exception): raise line

                    event_count += 1
                    parsed_results = self._parse_ndjson_line_comprehensive(line, event_count)
                    
                    for content_type, raw_content in parsed_results:
                        if raw_content is None: continue
                        
                        cleaned_chunk = self._clean_stream_chunk(raw_content)
                        
                        if cleaned_chunk and not self._is_duplicate_content(cleaned_chunk):
                            # 简化逻辑：直接发送所有有效内容，不做复杂的思考/回答判断
                            chunk = create_chat_completion_chunk(request_id, model_name, content=cleaned_chunk)
                            yield create_sse_data(chunk)
                            accumulated_content.append(cleaned_chunk)
                            self.last_content = cleaned_chunk

                full_response = "".join(accumulated_content)
                if full_response:
                    logger.info(f"完整响应（{event_count}个事件）: {full_response[:200]}...")
                else:
                    logger.warning(f"警告: 处理了{event_count}个事件但未提取到任何有效文本。")
                
                # 移除思考模式的复杂逻辑

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
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            raise HTTPException(status_code=400, detail="此端点当前仅支持流式响应 (stream=true)。")
            
    def _clean_stream_chunk(self, content: str) -> str:
        if not content:
            return ""
        
        # 保存原始内容用于调试
        original_content = content
        
        # 1. 只处理明显的垃圾内容，不做过度清理
        # 移除 "Start new chat" 和类似Base64的乱码
        content = content.replace("Start new chat", "")
        content = re.sub(r'[A-Za-z0-9+/=]{50,}\s*', '', content)
        
        # 2. 只在内容中间有过多空行时才清理，保留开头和结尾的空格
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # 3. 只移除结尾的空白，保留开头的空格（可能是缩进）
        content = content.rstrip()
        
        # 4. 如果清理后内容为空或只有空白，返回空字符串
        if not content or content.isspace():
            return ""

        return content
    
    def _is_duplicate_content(self, content: str) -> bool:
        """检查是否为重复内容"""
        if not content or len(content) < 3:
            return False
            
        # 检查是否与上一个内容完全相同
        if content == self.last_content:
            return True
            
        # 检查是否在缓冲区中已存在
        if content in self.content_buffer:
            return True
            
        # 添加到缓冲区（保持最近20个内容）
        self.content_buffer.append(content)
        if len(self.content_buffer) > 20:
            self.content_buffer.pop(0)
            
        return False
    
    # 移除不必要的复杂判断逻辑，简化处理流程

    def _prepare_headers(self, account: NotionAccount) -> Dict[str, str]:
        cookie_source = (account.cookie or "").strip()
        cookie_header = cookie_source if "=" in cookie_source else f"token_v2={cookie_source}"

        return {
            "Content-Type": "application/json", "Accept": "application/x-ndjson",
            "Cookie": cookie_header, "x-notion-space-id": account.space_id,
            "x-notion-active-user-header": account.user_id,
            "x-notion-client-version": account.client_version,
            "notion-audit-log-platform": "web", "Origin": "https://www.notion.so",
            "Referer": "https://www.notion.so/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        }

    def _normalize_block_id(self, block_id: str) -> str:
        if not block_id: return block_id
        b = block_id.replace("-", "").strip()
        if len(b) == 32 and re.fullmatch(r"[0-9a-fA-F]{32}", b):
            return f"{b[0:8]}-{b[8:12]}-{b[12:16]}-{b[16:20]}-{b[20:]}"
        return block_id

    def _prepare_payload(self, request_data: Dict[str, Any], thread_id: str, mapped_model: str, account: NotionAccount) -> Dict[str, Any]:
        req_block_id = request_data.get("notion_block_id") or account.block_id
        normalized_block_id = self._normalize_block_id(req_block_id) if req_block_id else None

        context_value: Dict[str, Any] = {
            "timezone": "Asia/Shanghai", "spaceId": account.space_id,
            "userId": account.user_id, "userEmail": account.user_email,
            "currentDatetime": datetime.now().astimezone().isoformat(),
        }
        if normalized_block_id:
            context_value["blockId"] = normalized_block_id

        config_value: Dict[str, Any]
        if mapped_model.startswith("vertex-"):
            context_value.update({"userName": f" {account.user_name}", "spaceName": f"{account.user_name}的 Notion", "surface": "ai_module"})
            config_value = { "type": "markdown-chat", "model": mapped_model, "useWebSearch": True }
        else:
            context_value.update({"userName": account.user_name, "surface": "workflows"})
            config_value = {"type": "workflow", "model": mapped_model, "useWebSearch": True}

        transcript = [
            {"id": str(uuid.uuid4()), "type": "config", "value": config_value},
            {"id": str(uuid.uuid4()), "type": "context", "value": context_value}
        ]
        for msg in request_data.get("messages", []):
            role, content = msg.get("role"), msg.get("content")
            if role == "user":
                transcript.append({"id": str(uuid.uuid4()), "type": "user", "value": [[content]], "userId": account.user_id, "createdAt": datetime.now().astimezone().isoformat()})
            elif role == "assistant":
                transcript.append({"id": str(uuid.uuid4()), "type": "agent-inference", "value": [{"type": "text", "content": content}]})

        payload = {"traceId": str(uuid.uuid4()), "spaceId": account.space_id, "transcript": transcript, "threadId": thread_id, "createThread": False, "isPartialTranscript": True, "asPatchResponse": True, "generateTitle": True, "saveAllThreadOperations": True, "threadType": config_value["type"]}
        return payload
    
    def _parse_ndjson_line_comprehensive(self, line: bytes, event_num: int) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        try:
            s = line.decode("utf-8", errors="ignore").strip()
            if not s: return results
            data = json.loads(s)
            
            if event_num <= 3: logger.debug(f"事件#{event_num} 完整数据: {json.dumps(data, ensure_ascii=False)[:500]}")
            
            if data.get("type") == "markdown-chat":
                content = data.get("value", "")
                if content: results.append(('direct', content))
            elif data.get("type") == "update":
                if "value" in data:
                    value = data["value"]
                    if isinstance(value, str) and value: results.append(('update', value))
                    elif isinstance(value, dict):
                        for key in ["content", "text", "message", "value"]:
                            if key in value:
                                content = value[key]
                                if isinstance(content, str) and content: results.append(('update', content)); break
            elif data.get("type") == "patch" and "v" in data:
                for op in data.get("v", []):
                    if not isinstance(op, dict): continue
                    op_type, val = op.get("o"), op.get("v")
                    if op_type in ("r", "s", "x") and isinstance(val, str) and val: results.append((op_type, val))
                    elif op_type == "a":
                        if isinstance(val, str) and val: results.append(('add', val))
                        elif isinstance(val, dict):
                            if val.get("type") == "text" and "content" in val and val["content"]: results.append(('add', val["content"]))
                            elif val.get("type") in ("markdown-chat", "text") and "value" in val and val["value"]: results.append(('add', val["value"]))
            elif data.get("type") == "record-map" and "recordMap" in data:
                if "thread_message" in data["recordMap"]:
                    for msg_data in data["recordMap"]["thread_message"].values():
                        step = msg_data.get("value", {}).get("value", {}).get("step", {})
                        if step:
                            content = self._extract_content_from_step(step)
                            if content: results.append(('record', content)); break
            elif "content" in data and isinstance(data["content"], str) and data["content"]: results.append(('root', data["content"]))
            elif "text" in data and isinstance(data["text"], str) and data["text"]: results.append(('root', data["text"]))
        except (json.JSONDecodeError, AttributeError): pass
        return results

    def _extract_content_from_step(self, step: Dict[str, Any]) -> Optional[str]:
        step_type = step.get("type")
        if step_type in ("markdown-chat", "text"): return step.get("value", "") or step.get("content", "")
        elif step_type == "agent-inference":
            for item in step.get("value", []):
                if isinstance(item, dict) and item.get("type") == "text": return item.get("content", "")
        if "value" in step and isinstance(step["value"], str): return step["value"]
        return None

    async def get_models(self) -> JSONResponse:
        model_data = {"object": "list", "data": [{"id": name, "object": "model", "created": int(time.time()), "owned_by": "lzA6"} for name in settings.KNOWN_MODELS]}
        return JSONResponse(content=model_data)

