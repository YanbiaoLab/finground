"""ADK budget reminders for Multi-KPI benchmark invocations."""

from __future__ import annotations

import math

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from finground.tools.submission import MAX_MULTI_KPI_RECORD_ROWS

FIRST_REMINDER_RATIO = 0.60
FINAL_WARNING_RATIO = 0.80

FIRST_REMINDER = f"""[LLM CALL BUDGET: 60% USED]
Before any more retrieval, immediately call record_multi_kpi_progress in batches of at most
{MAX_MULTI_KPI_RECORD_ROWS} rows with every selected source token and its page, row label, target
fiscal-year label, unit text, and scope. Do not calculate value; the tool normalizes it. Then query
pending_kpis and resolve them in grouped statement or note batches. Mark a KPI absent only after
checking its relevant statement or note; do not bulk-mark unseen KPIs absent. A printed dash/nil on
a matching row is explicit_zero; an absent row is missing and must never become zero."""

FINAL_WARNING = """[LLM CALL BUDGET: 80% USED]
Do not begin another search cycle. In this call, query the validated draft with
query_multi_kpi_progress(view="kpis"). The next model call must call submit_multi_kpi_extraction.
Pass kpis=[] to build the Ledger result from recorded evidence. Only if a final fact was not yet
recorded may you pass it using the full evidence format; never pass a calculated value. Do not start
new retrieval. Missing or ambiguous coverage is omitted; explicit_zero is retained as value 0. If
pending_kpis remain, the forced submission will be persisted as incomplete rather than counted as a
successful complete report."""


class MultiKpiExecutionGuardPlugin(BasePlugin):
    """Keep Multi-KPI work inside one invocation and enforce budget reminders."""

    def __init__(self, *, max_calls: int) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        super().__init__(name="finground_multi_kpi_execution_guard")
        self.call_count = 0
        self.prevented_early_stops = 0
        self.first_reminder_call = math.ceil(max_calls * FIRST_REMINDER_RATIO)
        self.final_warning_call = math.ceil(max_calls * FINAL_WARNING_RATIO)

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        self.call_count += 1
        message = None
        if self.call_count == self.first_reminder_call:
            message = FIRST_REMINDER
        elif self.call_count == self.final_warning_call:
            message = FINAL_WARNING
        if message is not None:
            llm_request.contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message)],
                )
            )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        """Replace a premature non-tool answer with a progress query tool call."""
        if (
            llm_response.get_function_calls()
            or llm_response.error_code is not None
            or llm_response.interrupted
            or llm_response.partial
        ):
            return None
        self.prevented_early_stops += 1
        guarded_response = llm_response.model_copy(deep=True)
        guarded_response.content = types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="query_multi_kpi_progress",
                        args={"view": "kpis"},
                    )
                )
            ],
        )
        guarded_response.partial = False
        return guarded_response
