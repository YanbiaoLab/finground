"""ADK-native LLM call metrics for benchmark invocations."""

from __future__ import annotations

from collections import Counter
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from finground.tools import MULTI_KPI_ALLOW_PARTIAL_STATE_KEY


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
            callback_context.state[MULTI_KPI_ALLOW_PARTIAL_STATE_KEY] = True
            llm_request.config = llm_request.config or types.GenerateContentConfig()
            for tool_group in llm_request.config.tools or []:
                declarations = tool_group.function_declarations or []
                tool_group.function_declarations = [
                    declaration
                    for declaration in declarations
                    if declaration.name == self.forced_tool_name
                ]
            llm_request.config.tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=[self.forced_tool_name],
                )
            )
        return None


class MultiKpiRunMetricsPlugin(BasePlugin):
    """Collect compact per-run diagnostics without retaining report payloads."""

    def __init__(self) -> None:
        super().__init__(name="finground_multi_kpi_run_metrics")
        self.prompt_token_total = 0
        self.prompt_token_max = 0
        self.candidate_token_total = 0
        self.total_token_count = 0
        self.tool_calls: Counter[str] = Counter()
        self.tool_statuses: Counter[str] = Counter()
        self.validation_error_count = 0
        self.retryable_error_calls = 0
        self.partial_success_calls = 0
        self.repeated_validation_error_calls = 0
        self.saved_kpi_rows = 0
        self.latest_pending_count: int | None = None
        self.tool_exception_count = 0
        self._previous_validation_signature: tuple[str, ...] | None = None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        usage = llm_response.usage_metadata
        if usage is None:
            return None
        prompt_tokens = int(usage.prompt_token_count or 0)
        self.prompt_token_total += prompt_tokens
        self.prompt_token_max = max(self.prompt_token_max, prompt_tokens)
        self.candidate_token_total += int(usage.candidates_token_count or 0)
        self.total_token_count += int(usage.total_token_count or 0)
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> dict | None:
        del tool_args, tool_context
        tool_name = tool.name
        self.tool_calls[tool_name] += 1
        status = str(result.get("status", "unknown"))
        self.tool_statuses[f"{tool_name}:{status}"] += 1
        if status == "partial_success":
            self.partial_success_calls += 1
        if status == "error" and result.get("retryable") is True:
            self.retryable_error_calls += 1

        validation_errors = result.get("validation_errors")
        if isinstance(validation_errors, list):
            self.validation_error_count += len(validation_errors)
            signature = tuple(
                sorted(
                    str(item.get("message", ""))
                    for item in validation_errors
                    if isinstance(item, dict)
                )
            )
            if signature and signature == self._previous_validation_signature:
                self.repeated_validation_error_calls += 1
            self._previous_validation_signature = signature or None
        else:
            self._previous_validation_signature = None

        self.saved_kpi_rows += int(result.get("added_kpi_count", 0) or 0)
        pending_count = result.get("pending_count")
        if isinstance(pending_count, int):
            self.latest_pending_count = pending_count
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict | None:
        del tool_args, tool_context, error
        self.tool_calls[tool.name] += 1
        self.tool_statuses[f"{tool.name}:exception"] += 1
        self.tool_exception_count += 1
        return None

    def snapshot(self) -> dict[str, Any]:
        """Return a stable JSON-serializable diagnostic summary."""
        return {
            "model_tokens": {
                "prompt_total": self.prompt_token_total,
                "prompt_max": self.prompt_token_max,
                "candidate_total": self.candidate_token_total,
                "total": self.total_token_count,
            },
            "tool_calls": dict(self.tool_calls),
            "tool_statuses": dict(self.tool_statuses),
            "validation_error_count": self.validation_error_count,
            "retryable_error_calls": self.retryable_error_calls,
            "partial_success_calls": self.partial_success_calls,
            "repeated_validation_error_calls": self.repeated_validation_error_calls,
            "saved_kpi_rows": self.saved_kpi_rows,
            "latest_pending_count": self.latest_pending_count,
            "tool_exception_count": self.tool_exception_count,
        }
