"""Shared ADK model configuration for FinGround agents."""

from __future__ import annotations

from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLlm

from finground.config import load_settings

SETTINGS = load_settings()


def create_adk_model(model_name: str, *, json_output: bool = False) -> str | BaseLlm:
    """Resolve provider-specific model names into an ADK model implementation."""
    if model_name.startswith("deepseek-"):
        kwargs = {"response_format": {"type": "json_object"}} if json_output else {}
        return LiteLlm(model=f"deepseek/{model_name}", drop_params=True, **kwargs)
    if model_name.casefold().startswith("qwen"):
        kwargs = {"response_format": {"type": "json_object"}} if json_output else {}
        return LiteLlm(
            model=f"openai/{model_name}",
            api_base=SETTINGS.vllm_base_url,
            api_key=SETTINGS.vllm_api_key,
            drop_params=True,
            tool_choice="required",
            parallel_tool_calls=False,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            **kwargs,
        )
    return model_name


ADK_MODEL = create_adk_model(SETTINGS.model)
