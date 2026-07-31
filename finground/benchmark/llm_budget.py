"""ADK routing and budget guardrails for the Multi-KPI agent hierarchy."""

from __future__ import annotations

import json
import math
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from finground.agents.kpi_specialists import (
    COMMON_TASK_AGENT_NAME,
    KPI_DISPATCH_TOOL_NAME,
    MULTI_KPI_COORDINATOR_NAME,
)
from finground.kpis import KPI_KEYS
from finground.sec_facts import SEC_FACTS_STATE_KEY
from finground.tools import (
    MULTI_KPI_ALLOW_PARTIAL_STATE_KEY,
    MULTI_KPI_PREPARED_STATE_KEY,
    MULTI_KPI_REQUESTED_STATE_KEY,
    MULTI_KPI_WORK_RECORD_STATE_KEY,
    REPORT_STATE_KEY,
)

FIRST_REMINDER_RATIO = 0.60
FINAL_WARNING_RATIO = 0.80

FIRST_REMINDER = """[MULTI-AGENT BUDGET: 60% USED]
Continue delegating only pending KPI specialists. Each specialist must persist exactly one status;
do not repeat completed specialists or move their evidence into the coordinator context."""

FINAL_WARNING = """[MULTI-AGENT BUDGET: 80% USED]
Prioritize pending KPI specialists, accept grounded absent or ambiguous decisions after their
bounded source checks, then ask manage_report_workflow to audit and finalize the state-backed
result."""


def _restrict_tools(llm_request: LlmRequest, allowed_names: list[str]) -> None:
    llm_request.config = llm_request.config or types.GenerateContentConfig()
    for tool_group in llm_request.config.tools or []:
        declarations = tool_group.function_declarations or []
        tool_group.function_declarations = [
            declaration for declaration in declarations if declaration.name in allowed_names
        ]
    llm_request.config.tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.ANY,
            allowed_function_names=allowed_names,
        )
    )


def _coverage_from_state(context: CallbackContext | ToolContext) -> tuple[int, list[str]]:
    report = context.state.get(REPORT_STATE_KEY, {})
    work_record = context.state.get(MULTI_KPI_WORK_RECORD_STATE_KEY, {})
    report_year = report.get("year") if isinstance(report, dict) else None
    rows = work_record.get("kpis", []) if isinstance(work_record, dict) else []
    covered = {
        row.get("kpi")
        for row in rows
        if isinstance(row, dict) and row.get("fiscal_year") == report_year
    }
    sec_facts = context.state.get(SEC_FACTS_STATE_KEY, {})
    structured_values = sec_facts.get("values", {}) if isinstance(sec_facts, dict) else {}
    if isinstance(structured_values, dict):
        covered.update(kpi for kpi in structured_values if kpi in KPI_KEYS)
    requested = context.state.get(MULTI_KPI_REQUESTED_STATE_KEY)
    scope = (
        [kpi for kpi in KPI_KEYS if kpi in requested]
        if isinstance(requested, list) and requested
        else list(KPI_KEYS)
    )
    pending = [kpi for kpi in scope if kpi not in covered]
    return len(covered), pending


def _agent_name(context: CallbackContext | ToolContext | None) -> str | None:
    return getattr(context, "agent_name", None) if context is not None else None


class MultiKpiExecutionGuardPlugin(BasePlugin):
    """Route the coordinator while leaving each isolated specialist autonomous."""

    def __init__(self, *, max_calls: int) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        super().__init__(name="finground_multi_kpi_execution_guard")
        self.max_calls = max_calls
        self.call_count = 0
        self.prevented_early_stops = 0
        self.first_reminder_call = math.ceil(max_calls * FIRST_REMINDER_RATIO)
        self.final_warning_call = math.ceil(max_calls * FINAL_WARNING_RATIO)
        self._first_reminder_sent = False
        self._final_warning_sent = False

    @staticmethod
    def _prepared(context: CallbackContext | ToolContext) -> bool:
        return isinstance(context.state.get(MULTI_KPI_PREPARED_STATE_KEY), dict)

    def _allowed_coordinator_tools(
        self,
        callback_context: CallbackContext,
        pending_kpis: list[str],
    ) -> list[str]:
        if self.call_count >= self.max_calls:
            callback_context.state[MULTI_KPI_ALLOW_PARTIAL_STATE_KEY] = True
            return [COMMON_TASK_AGENT_NAME]
        if not self._prepared(callback_context):
            return [COMMON_TASK_AGENT_NAME]
        if pending_kpis:
            return [KPI_DISPATCH_TOOL_NAME]
        return [COMMON_TASK_AGENT_NAME]

    def _routing_message(
        self,
        callback_context: CallbackContext,
        pending_kpis: list[str],
    ) -> str:
        report = callback_context.state.get(REPORT_STATE_KEY, {})
        report_id = report.get("report_id") if isinstance(report, dict) else None
        return "\n".join(
            [
                "[COORDINATOR ROUTING STATE — AUTHORITATIVE]",
                f"total_model_calls={self.call_count}/{self.max_calls}; report={report_id}; "
                f"prepared={self._prepared(callback_context)}; "
                f"coverage={len(KPI_KEYS) - len(pending_kpis)}/{len(KPI_KEYS)}",
                f"pending_kpis={json.dumps(pending_kpis)}",
                (
                    f"next_action={COMMON_TASK_AGENT_NAME} finalize available state"
                    if self.call_count >= self.max_calls
                    else (
                        f"next_action={COMMON_TASK_AGENT_NAME} prepare"
                        if not self._prepared(callback_context)
                        else (
                            f"next_action={COMMON_TASK_AGENT_NAME} finalize"
                            if not pending_kpis
                            else "next_action=delegate one pending KPI specialist"
                        )
                    )
                ),
            ]
        )

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        self.call_count += 1
        if callback_context is None or _agent_name(callback_context) != MULTI_KPI_COORDINATOR_NAME:
            return None

        _coverage_count, pending_kpis = _coverage_from_state(callback_context)
        _restrict_tools(
            llm_request,
            self._allowed_coordinator_tools(callback_context, pending_kpis),
        )
        llm_request.contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=self._routing_message(callback_context, pending_kpis))
                ],
            )
        )
        if self.call_count >= self.first_reminder_call and not self._first_reminder_sent:
            self._first_reminder_sent = True
            llm_request.contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=FIRST_REMINDER)],
                )
            )
        if self.call_count >= self.final_warning_call and not self._final_warning_sent:
            self._final_warning_sent = True
            llm_request.contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=FINAL_WARNING)],
                )
            )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict | None:
        agent_name = _agent_name(tool_context)
        if agent_name == MULTI_KPI_COORDINATOR_NAME and tool.name == KPI_DISPATCH_TOOL_NAME:
            if not self._prepared(tool_context):
                return {
                    "status": "error",
                    "retryable": True,
                    "error": "report preparation is required before KPI delegation",
                    "next_action": COMMON_TASK_AGENT_NAME,
                }
            _coverage_count, pending_kpis = _coverage_from_state(tool_context)
            requested = tool_args.get("kpis")
            invalid = (
                not isinstance(requested, list)
                or not requested
                or any(kpi not in pending_kpis for kpi in requested)
            )
            if invalid:
                return {
                    "status": "error",
                    "retryable": True,
                    "error": "dispatcher kpis must be the currently pending canonical KPI keys",
                    "pending_kpis": pending_kpis,
                }
        return None

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        """Replace a premature coordinator answer with a state-backed delegation."""
        if (
            _agent_name(callback_context) != MULTI_KPI_COORDINATOR_NAME
            or llm_response.get_function_calls()
            or llm_response.error_code is not None
            or llm_response.interrupted
            or llm_response.partial
        ):
            return None

        _coverage_count, pending_kpis = _coverage_from_state(callback_context)
        if self.call_count >= self.max_calls:
            callback_context.state[MULTI_KPI_ALLOW_PARTIAL_STATE_KEY] = True
        if not self._prepared(callback_context):
            tool_name = COMMON_TASK_AGENT_NAME
            request = "Prepare report metadata and primary-statement source indexes."
        elif pending_kpis and self.call_count < self.max_calls:
            tool_name = KPI_DISPATCH_TOOL_NAME
            request = "Find, validate, and checkpoint every supplied pending KPI."
        else:
            tool_name = COMMON_TASK_AGENT_NAME
            request = "Finalize and submit the state-backed extraction."

        self.prevented_early_stops += 1
        guarded_response = llm_response.model_copy(deep=True)
        guarded_response.content = types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name=tool_name,
                        args=(
                            {"kpis": pending_kpis, "request": request}
                            if tool_name == KPI_DISPATCH_TOOL_NAME
                            else {"request": request}
                        ),
                    )
                )
            ],
        )
        guarded_response.partial = False
        return guarded_response
