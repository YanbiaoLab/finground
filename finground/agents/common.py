"""Shared ADK model configuration for FinGround agents."""

from __future__ import annotations

from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLlm

from finground.config import load_settings

SETTINGS = load_settings()


def create_adk_model(
    model_name: str,
    *,
    json_output: bool = False,
    json_schema: dict | None = None,
) -> str | BaseLlm:
    """Resolve provider-specific model names into an ADK model implementation."""
    response_format = (
        {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": json_schema,
                "strict": True,
            },
        }
        if json_schema is not None
        else {"type": "json_object"}
    )
    kwargs = {"response_format": response_format} if json_output else {}
    if model_name.startswith("deepseek-"):
        return LiteLlm(model=f"deepseek/{model_name}", drop_params=True, **kwargs)
    if model_name.casefold().startswith("qwen"):
        tool_kwargs = (
            {}
            if json_output
            else {
                "tool_choice": "required",
                "parallel_tool_calls": False,
            }
        )
        return LiteLlm(
            model=f"openai/{model_name}",
            api_base=SETTINGS.vllm_base_url,
            api_key=SETTINGS.vllm_api_key,
            drop_params=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            **tool_kwargs,
            **kwargs,
        )
    return model_name


ADK_MODEL = create_adk_model(SETTINGS.model)
