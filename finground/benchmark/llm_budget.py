"""ADK budget reminders for Multi-KPI benchmark invocations."""

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

from finground.kpis import KPI_KEYS
from finground.tools import MULTI_KPI_WORK_RECORD_STATE_KEY, REPORT_STATE_KEY
from finground.tools.submission import MAX_MULTI_KPI_RECORD_ROWS

FIRST_REMINDER_RATIO = 0.60
FINAL_WARNING_RATIO = 0.80
RETRIEVAL_TOOL_NAMES = {
    "get_report_info",
    "read_report_pages",
    "search_report",
}
DEDUPLICATED_TOOL_NAMES = {
    *RETRIEVAL_TOOL_NAMES,
    "query_multi_kpi_progress",
    "record_multi_kpi_progress",
}
CLOSURE_TOOL_NAMES = [
    "record_multi_kpi_progress",
    "query_multi_kpi_progress",
    "submit_multi_kpi_extraction",
]
RECOVERY_TOOL_NAMES = [
    "search_report",
    "record_multi_kpi_progress",
    "query_multi_kpi_progress",
]

FIRST_REMINDER = f"""[LLM CALL BUDGET: 60% USED]
Finish the active source checkpoint now with record_multi_kpi_progress. Record all valid rows in
batches of at most {MAX_MULTI_KPI_RECORD_ROWS}; a partial_success already saved valid rows, so retry
only rejected rows. Before any more retrieval, confirm that income, balance sheet, and cash-flow
batches were all processed. Then query pending_kpis, use grouped note cycles, and move to coverage
closure. A printed dash/nil is explicit_zero; no matching row after the planned source check is
absent; relevant but unresolved evidence is ambiguous."""

FINAL_WARNING_TEMPLATE = """[LLM CALL BUDGET: 80% USED — CLOSURE ONLY]
All remaining calls are closure-only: no new retrieval. Record grounded absent or ambiguous
statuses for remaining pending KPIs in batches of at most {max_rows}, query
query_multi_kpi_progress(view="kpis"), and submit kpis=[] no later than forced call {max_calls}.
Do not resend valid rows from partial_success. Do not calculate values or turn missing rows into
zero."""


def _state_backed_submission_args(callback_context: CallbackContext) -> dict:
    report = callback_context.state.get(REPORT_STATE_KEY, {})
    work_record = callback_context.state.get(MULTI_KPI_WORK_RECORD_STATE_KEY, {})
    if not isinstance(report, dict):
        report = {}
    if not isinstance(work_record, dict):
        work_record = {}
    return {
        "ticker": report.get("ticker", ""),
        "reporting_currency": work_record.get("reporting_currency"),
        "units_note": work_record.get("units_note"),
        "kpis": [],
    }


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


def _coverage_from_state(callback_context: CallbackContext) -> tuple[int, list[str]]:
    report = callback_context.state.get(REPORT_STATE_KEY, {})
    work_record = callback_context.state.get(MULTI_KPI_WORK_RECORD_STATE_KEY, {})
    report_year = report.get("year") if isinstance(report, dict) else None
    rows = work_record.get("kpis", []) if isinstance(work_record, dict) else []
    covered = {
        row.get("kpi")
        for row in rows
        if isinstance(row, dict) and row.get("fiscal_year") == report_year
    }
    pending = [kpi for kpi in KPI_KEYS if kpi not in covered]
    return len(covered), pending


class MultiKpiExecutionGuardPlugin(BasePlugin):
    """Keep Multi-KPI work inside one invocation and enforce budget reminders."""

    def __init__(self, *, max_calls: int, max_searches: int = 7) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        if max_searches < 1:
            raise ValueError("max_searches must be at least 1")
        super().__init__(name="finground_multi_kpi_execution_guard")
        self.max_calls = max_calls
        self.max_searches = max_searches
        self.call_count = 0
        self.prevented_early_stops = 0
        self.first_reminder_call = math.ceil(max_calls * FIRST_REMINDER_RATIO)
        self.final_warning_call = math.ceil(max_calls * FINAL_WARNING_RATIO)
        self._report_info_done = False
        self._awaiting_checkpoint_pages: list[int] = []
        self._search_requires_read = False
        self._coverage_count = 0
        self._retrieval_coverage: dict[str, int] = {}
        self._repair_queue: list[dict[str, Any]] = []
        self._repair_attempts: dict[str, int] = {}
        self._deferred_repairs: list[dict[str, Any]] = []
        self._duplicate_recovery = False
        self._blocked_tool_name: str | None = None
        self._search_count = 0

    @staticmethod
    def _retrieval_signature(tool_name: str, tool_args: dict[str, Any]) -> str:
        return f"{tool_name}:{json.dumps(tool_args, sort_keys=True, default=str)}"

    def _workflow_phase(
        self, callback_context: CallbackContext
    ) -> tuple[str, list[str] | None, list[str]]:
        coverage_count, pending_kpis = _coverage_from_state(callback_context)
        self._coverage_count = max(self._coverage_count, coverage_count)
        if not pending_kpis:
            return "submit", ["submit_multi_kpi_extraction"], pending_kpis
        if self.call_count >= self.final_warning_call:
            return "closure", CLOSURE_TOOL_NAMES, pending_kpis
        if not self._report_info_done:
            return "metadata", ["get_report_info"], pending_kpis
        if self._duplicate_recovery:
            recovery_tools = [
                name for name in RECOVERY_TOOL_NAMES if name != self._blocked_tool_name
            ]
            return "duplicate_recovery", recovery_tools, pending_kpis
        if self._awaiting_checkpoint_pages:
            return (
                "repair" if self._repair_queue else "checkpoint",
                ["record_multi_kpi_progress"],
                pending_kpis,
            )
        if self._search_requires_read:
            return "source_read", ["read_report_pages"], pending_kpis
        if self._deferred_repairs:
            return "repair_exhausted", RECOVERY_TOOL_NAMES, pending_kpis
        return "discovery", None, pending_kpis

    def _workflow_message(
        self,
        callback_context: CallbackContext,
        phase: str,
        pending_kpis: list[str],
    ) -> str:
        report = callback_context.state.get(REPORT_STATE_KEY, {})
        report_id = report.get("report_id") if isinstance(report, dict) else None
        details = [
            "[WORKFLOW STATE — AUTHORITATIVE]",
            f"call={self.call_count}/{self.max_calls}; phase={phase}; report={report_id}; "
            f"coverage={len(KPI_KEYS) - len(pending_kpis)}/{len(KPI_KEYS)}",
            f"pending_kpis={json.dumps(pending_kpis)}",
        ]
        if self._awaiting_checkpoint_pages:
            details.append(
                f"active_source_pages={json.dumps(self._awaiting_checkpoint_pages)}; "
                "next_action=record_multi_kpi_progress"
            )
        if self._repair_queue:
            details.append(
                "repair_queue="
                f"{json.dumps(self._repair_queue, ensure_ascii=True, separators=(',', ':'))}; "
                "resubmit only these rejected KPI rows"
            )
        if self._deferred_repairs:
            details.append(
                "repair_exhausted="
                f"{json.dumps(self._deferred_repairs, ensure_ascii=True, separators=(',', ':'))}; "
                "do not retry the same evidence again; choose a new source or record ambiguous "
                "after the bounded source check"
            )
        if phase == "source_read":
            details.append("next_action=read_report_pages using the current search results")
        elif phase == "closure":
            details.append(
                "retrieval is disabled; close pending coverage in batches, query once, then submit"
            )
        elif phase == "submit":
            details.append("coverage is complete; submit kpis=[] now")
        return "\n".join(details)

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        self.call_count += 1
        if callback_context is not None:
            phase, allowed_names, pending_kpis = self._workflow_phase(callback_context)
            if allowed_names is not None:
                _restrict_tools(llm_request, allowed_names)
            llm_request.contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=self._workflow_message(
                                callback_context,
                                phase,
                                pending_kpis,
                            )
                        )
                    ],
                )
            )
        message = None
        if self.call_count == self.first_reminder_call:
            message = FIRST_REMINDER
        elif self.call_count == self.final_warning_call:
            message = FINAL_WARNING_TEMPLATE.format(
                max_rows=MAX_MULTI_KPI_RECORD_ROWS,
                max_calls=self.max_calls,
            )
        if message is not None:
            llm_request.contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message)],
                )
            )
        if self.call_count >= self.max_calls:
            submit_args = json.dumps(
                _state_backed_submission_args(callback_context),
                ensure_ascii=True,
            )
            llm_request.contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=(
                                "[FINAL CALL] Call submit_multi_kpi_extraction now with exactly "
                                f"these arguments: {submit_args}"
                            )
                        )
                    ],
                )
            )
        return None

    def _track_repair_queue(self, result: dict) -> None:
        repair_queue = result.get("repair_queue")
        if not isinstance(repair_queue, list) or not repair_queue:
            self._repair_queue = []
            return
        signature = json.dumps(repair_queue, sort_keys=True, default=str)
        attempts = self._repair_attempts.get(signature, 0) + 1
        self._repair_attempts[signature] = attempts
        if attempts >= 2:
            self._deferred_repairs = repair_queue
            self._repair_queue = []
            self._awaiting_checkpoint_pages = []
        else:
            self._repair_queue = repair_queue

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict | None:
        tool_name = tool.name
        if tool_name == "search_report" and self._search_count >= self.max_searches:
            self._duplicate_recovery = True
            self._blocked_tool_name = tool_name
            return {
                "status": "error",
                "retryable": False,
                "error": "search budget exhausted",
                "blocked_tool": tool_name,
                "search_count": self._search_count,
                "next_action": "query progress or record grounded coverage",
            }
        if tool_name == "get_report_info" and self._report_info_done:
            return {
                "status": "error",
                "retryable": True,
                "error": "duplicate retrieval blocked",
                "blocked_tool": tool_name,
                "next_action": (
                    "record_multi_kpi_progress"
                    if self._awaiting_checkpoint_pages
                    else "continue from the existing report metadata"
                ),
            }
        if tool_name in DEDUPLICATED_TOOL_NAMES:
            signature = self._retrieval_signature(tool_name, tool_args)
            previous_coverage = self._retrieval_coverage.get(signature)
            if previous_coverage is not None and previous_coverage >= self._coverage_count:
                self._duplicate_recovery = True
                self._blocked_tool_name = tool_name
                if tool_name == "read_report_pages":
                    self._search_requires_read = False
                elif tool_name == "record_multi_kpi_progress":
                    self._awaiting_checkpoint_pages = []
                    self._repair_queue = []
                return {
                    "status": "error",
                    "retryable": True,
                    "error": (
                        "duplicate retrieval blocked"
                        if tool_name in RETRIEVAL_TOOL_NAMES
                        else "duplicate progress action blocked"
                    ),
                    "blocked_tool": tool_name,
                    "next_action": (
                        "record_multi_kpi_progress"
                        if self._awaiting_checkpoint_pages
                        else "choose a new source or query_multi_kpi_progress"
                    ),
                }
        if tool_name in RETRIEVAL_TOOL_NAMES:
            if self.call_count >= self.final_warning_call:
                return {
                    "status": "error",
                    "retryable": True,
                    "error": "retrieval disabled during closure",
                    "blocked_tool": tool_name,
                    "next_action": "record pending coverage, query progress, then submit",
                }
            if self._awaiting_checkpoint_pages:
                return {
                    "status": "error",
                    "retryable": True,
                    "error": "active retrieval requires checkpoint",
                    "blocked_tool": tool_name,
                    "active_source_pages": self._awaiting_checkpoint_pages,
                    "next_action": "record_multi_kpi_progress",
                }
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> dict | None:
        del tool_context
        tool_name = tool.name
        status = result.get("status")
        if status not in {"success", "partial_success"}:
            if tool_name == "record_multi_kpi_progress":
                self._track_repair_queue(result)
            return None

        coverage_count = result.get("coverage_count")
        previous_coverage_count = self._coverage_count
        if isinstance(coverage_count, int):
            self._coverage_count = max(self._coverage_count, coverage_count)
        if tool_name == "get_report_info":
            self._report_info_done = True
        elif tool_name == "search_report":
            self._search_count += 1
            signature = self._retrieval_signature(tool_name, tool_args)
            self._retrieval_coverage[signature] = self._coverage_count
            self._search_requires_read = bool(result.get("results"))
            self._duplicate_recovery = False
            self._blocked_tool_name = None
        elif tool_name == "read_report_pages":
            signature = self._retrieval_signature(tool_name, tool_args)
            self._retrieval_coverage[signature] = self._coverage_count
            pages = result.get("pages")
            self._awaiting_checkpoint_pages = [
                int(page["page"])
                for page in pages or []
                if isinstance(page, dict) and isinstance(page.get("page"), int)
            ]
            self._search_requires_read = False
            self._deferred_repairs = []
            self._duplicate_recovery = False
            self._blocked_tool_name = None
        elif tool_name == "record_multi_kpi_progress":
            signature = self._retrieval_signature(tool_name, tool_args)
            self._retrieval_coverage[signature] = self._coverage_count
            self._track_repair_queue(result)
            if not self._repair_queue:
                self._awaiting_checkpoint_pages = []
            if self._coverage_count > previous_coverage_count:
                self._duplicate_recovery = False
                self._blocked_tool_name = None
        elif tool_name == "query_multi_kpi_progress":
            signature = self._retrieval_signature(tool_name, tool_args)
            self._retrieval_coverage[signature] = self._coverage_count
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        """Replace a premature non-tool answer with a safe state-backed tool call."""
        if (
            llm_response.get_function_calls()
            or llm_response.error_code is not None
            or llm_response.interrupted
            or llm_response.partial
        ):
            return None
        self.prevented_early_stops += 1
        tool_name = "query_multi_kpi_progress"
        tool_args: dict = {"view": "kpis"}
        if self.call_count >= self.max_calls:
            tool_name = "submit_multi_kpi_extraction"
            tool_args = _state_backed_submission_args(callback_context)
        guarded_response = llm_response.model_copy(deep=True)
        guarded_response.content = types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name=tool_name,
                        args=tool_args,
                    )
                )
            ],
        )
        guarded_response.partial = False
        return guarded_response
