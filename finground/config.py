"""Environment-backed runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from google.adk.models.lite_llm import LiteLlm


def required_env(name: str) -> str:
    """Return a non-empty environment variable or fail with a clear message."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable {name} is not set")
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
            base_url=required_env("FINGROUND_MODEL_BASE_URL").rstrip("/"),
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
