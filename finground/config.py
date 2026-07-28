"""Runtime configuration loaded exclusively from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model: str = "qwen36-27b-fp8"
    vllm_base_url: str = "http://60.171.65.125:30845/v1"
    vllm_api_key: str = "EMPTY"


def load_settings() -> Settings:
    return Settings(
        model=os.getenv("FINGROUND_MODEL", "qwen36-27b-fp8"),
        vllm_base_url=os.getenv("FINGROUND_VLLM_BASE_URL", "http://60.171.65.125:30845/v1"),
        vllm_api_key=os.getenv("FINGROUND_VLLM_API_KEY", "EMPTY"),
    )
