# app/core/config.py
import os
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class NotionAccount(BaseModel):
    """用于封装单个 Notion 账号凭证的 Pydantic 模型"""
    cookie: str
    space_id: str
    user_id: str
    user_name: Optional[str] = "Notion User"
    user_email: Optional[str] = "notion@example.com"
    block_id: Optional[str] = None
    client_version: str

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra="ignore"
    )

    APP_NAME: str = "notion-2api"
    APP_VERSION: str = "4.1.0-Rotation"
    DESCRIPTION: str = "一个将 Notion AI 转换为兼容 OpenAI 格式 API 的高性能代理，支持多账号轮询。"

    API_MASTER_KEY: Optional[str] = None

    # --- 轮询和通用配置 ---
    NOTION_ACCOUNTS_NUM: int = Field(1, ge=1)
    NOTION_CLIENT_VERSION: str = "23.13.20251011.2037"

    # --- 存储加载后的账号列表 ---
    ACCOUNTS: List[NotionAccount] = []

    API_REQUEST_TIMEOUT: int = 180
    NGINX_PORT: int = 8088

    # --- 模型配置 (保持不变) ---
    DEFAULT_MODEL: str = "claude-sonnet-4.5"
    KNOWN_MODELS: List[str] = [
        "claude-sonnet-4.5", "gpt-5", "claude-opus-4.1",
        "gemini-2.5-flash（未修复，不可用）", "gemini-2.5-pro（未修复，不可用）", "gpt-4.1"
    ]
    MODEL_MAP: dict = {
        "claude-sonnet-4.5": "anthropic-sonnet-alt",
        "gpt-5": "openai-turbo",
        "claude-opus-4.1": "anthropic-opus-4.1",
        "gemini-2.5-flash（未修复，不可用）": "vertex-gemini-2.5-flash",
        "gemini-2.5-pro（未修复，不可用）": "vertex-gemini-2.5-pro",
        "gpt-4.1": "openai-gpt-4.1"
    }

    def __init__(self, **values):
        super().__init__(**values)
        self._load_accounts()

    def _load_accounts(self):
        """从环境变量加载多个 Notion 账号"""
        for i in range(1, self.NOTION_ACCOUNTS_NUM + 1):
            cookie = os.getenv(f"NOTION_COOKIE_{i}")
            space_id = os.getenv(f"NOTION_SPACE_ID_{i}")
            user_id = os.getenv(f"NOTION_USER_ID_{i}")

            if not all([cookie, space_id, user_id]):
                print(f"警告: 账号 {i} 的凭证不完整 (COOKIE, SPACE_ID, USER_ID 必须全部提供)，已跳过。")
                continue

            account = NotionAccount(
                cookie=cookie,
                space_id=space_id,
                user_id=user_id,
                user_name=os.getenv(f"NOTION_USER_NAME_{i}"),
                user_email=os.getenv(f"NOTION_USER_EMAIL_{i}"),
                block_id=os.getenv(f"NOTION_BLOCK_ID_{i}"),
                client_version=self.NOTION_CLIENT_VERSION
            )
            self.ACCOUNTS.append(account)
        
        if not self.ACCOUNTS:
            raise ValueError("配置错误: 未能成功加载任何 Notion 账号。请检查 .env 文件和 NOTION_ACCOUNTS_NUM 设置。")
        
        print(f"成功加载 {len(self.ACCOUNTS)} 个 Notion 账号。")

settings = Settings()

