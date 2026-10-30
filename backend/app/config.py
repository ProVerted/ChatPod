"""
Central place for env-based settings. Nothing here should ever be a
hardcoded secret, URL, or personal file path — that was one of the
original project's biggest problems.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider config (used by summarizer.py, and later qa.py)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"


settings = Settings()
