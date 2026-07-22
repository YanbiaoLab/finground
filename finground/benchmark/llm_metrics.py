"""ADK-native LLM call metrics for benchmark invocations."""

from __future__ import annotations

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types


class LlmCallCounterPlugin(BasePlugin):
    """Count model requests attempted during one benchmark invocation."""

    def __init__(
        self,
        *,
        max_calls: int | None = None,
        force_tool_at_call: int | None = None,
        forced_tool_name: str | None = None,
    ) -> None:
        super().__init__(name="finground_llm_call_counter")
        if (force_tool_at_call is None) != (forced_tool_name is None):
            raise ValueError("force_tool_at_call and forced_tool_name must be provided together")
        self.count = 0
        self.max_calls = max_calls
        self.force_tool_at_call = force_tool_at_call
        self.forced_tool_name = forced_tool_name

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        if self.max_calls is None or self.count < self.max_calls:
            self.count += 1
        if (
            self.force_tool_at_call is not None
            and self.count >= self.force_tool_at_call
            and self.forced_tool_name is not None
        ):
            llm_request.config = llm_request.config or types.GenerateContentConfig()
            llm_request.config.tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=[self.forced_tool_name],
                )
            )
        return None
