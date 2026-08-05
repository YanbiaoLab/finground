"""Environment-backed runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm

PROJECT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_project_env(env_file: Path = PROJECT_ENV_FILE) -> bool:
    """Load project-local variables without overriding the process environment."""
    return load_dotenv(dotenv_path=env_file, override=False)


load_project_env()


def required_env(name: str) -> str:
    """Return a non-empty environment variable or fail with a clear message."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable {name} is not set")
    return value


def required_service_url(name: str) -> str:
    """Return a real HTTP(S) service URL rather than an example placeholder."""
    value = required_env(name).rstrip("/")
    parsed = urlparse(value)
    hostname = (parsed.hostname or "").casefold()
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise RuntimeError(f"environment variable {name} must be an absolute HTTP(S) URL")
    if hostname == "example.com" or hostname.endswith(".example.com"):
        raise RuntimeError(
            f"environment variable {name} still contains the example placeholder {hostname}"
        )
    return value


@dataclass(frozen=True)
class ModelConfig:
    """Connection settings for the OpenAI-compatible agent model."""

    name: str
    base_url: str
    api_key: str = field(repr=False)

    @classmethod
    def from_env(cls) -> ModelConfig:
        return cls(
            name=required_env("FINGROUND_MODEL"),
            base_url=required_service_url("FINGROUND_MODEL_BASE_URL"),
            api_key=required_env("FINGROUND_MODEL_API_KEY"),
        )

    def create_model(self) -> LiteLlm:
        """Create an ADK LiteLLM model without exposing credentials in source."""
        return LiteLlm(
            model=self.name,
            api_base=self.base_url,
            api_key=self.api_key,
        )


def create_agent_model() -> LiteLlm:
    """Create one independently configured model instance for an agent."""
    return ModelConfig.from_env().create_model()
