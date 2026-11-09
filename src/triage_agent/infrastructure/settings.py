"""Configuration from environment variables. Secrets never have defaults in code."""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"

    database_url: str = "sqlite:///./triage.db"

    # "label:key" pairs, comma separated. The label becomes the audit actor (CREATEDBY).
    api_keys: SecretStr = SecretStr("")

    triage_agent: Literal["claude", "keyword"] = "claude"
    prompt_version: str = "v2"
    prompts_dir: Path = Path("prompts/triage")
    knowledge_base_path: Path = Path("data/knowledge_base.json")
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # Claude API. The key itself is read by the SDK from ANTHROPIC_API_KEY.
    claude_model: str = "claude-opus-5-5"
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    claude_max_tokens: int = 16000
    claude_timeout_seconds: float = 60.0
    claude_max_retries: int = Field(default=2, ge=0, le=5)
    claude_fallbacks_enabled: bool = True
    agent_max_tool_rounds: int = Field(default=3, ge=0, le=10)

    # USD per million tokens, used to report cost per ticket. Check current pricing.
    price_input_per_mtok: Decimal = Decimal("4.00")
    price_output_per_mtok: Decimal = Decimal("20.00")

    circuit_failure_threshold: int = Field(default=5, ge=1)
    circuit_reset_seconds: float = Field(default=30.0, gt=0)

    def parsed_api_keys(self) -> dict[str, str]:
        """Return {key: label}. Malformed entries are ignored (deny by default)."""
        keys: dict[str, str] = {}
        for entry in self.api_keys.get_secret_value().split(","):
            label, sep, key = entry.strip().partition(":")
            if sep and label and len(key) >= 16:
                keys[key] = label
        return keys


@lru_cache
def get_settings() -> Settings:
    return Settings()
