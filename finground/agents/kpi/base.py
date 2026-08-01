"""Mechanical ADK runtime support for autonomous KPI agent modules.

This module deliberately owns no KPI definitions, evidence policy, or prompts.
Those decisions belong to the 31 specialist modules so they can diverge freely.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from google.adk.agents import Agent
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

from finground.agents.common import ADK_MODEL
from finground.tools import (
    MULTI_KPI_WORK_RECORD_STATE_KEY,
    REPORT_STATE_KEY,
    find_kpi_source_candidates,
    read_report_pages,
    record_multi_kpi_progress,
    record_multi_kpi_progress_tool,
    search_kpi_report,
)
from finground.tools.structured import JsonSchemaFunctionTool

KPI_AGENT_NAME_PREFIX = "extract_"
KPI_SPECIALIST_SEARCH_LIMIT = 2
KPI_SPECIALIST_MODEL_TURN_LIMIT = 6


def kpi_agent_name(kpi: str) -> str:
    """Return the stable ADK name for one KPI specialist."""
    return f"{KPI_AGENT_NAME_PREFIX}{kpi}"


def _specialized_record_tool(kpi: str) -> JsonSchemaFunctionTool:
    schema = deepcopy(record_multi_kpi_progress_tool._get_declaration().parameters_json_schema)
    kpi_schema = schema["properties"]["kpis"]["items"]["properties"]["kpi"]
    kpi_schema.clear()
    kpi_schema.update(
        {
            "type": "string",
            "enum": [kpi],
            "description": f"This specialist may record only the canonical KPI {kpi}.",
        }
    )
    return JsonSchemaFunctionTool(record_multi_kpi_progress, parameters_json_schema=schema)


def _candidate_tool(kpi: str):
    def find_candidates(tool_context: ToolContext) -> dict:
        return find_kpi_source_candidates(kpi, tool_context)

    find_candidates.__name__ = f"find_{kpi}_candidates"
    find_candidates.__doc__ = (
        f"Return pre-indexed primary-statement source cells relevant only to {kpi}."
    )
    return find_candidates


def _search_tool(kpi: str):
    def search_report(
        query: str,
        phrases: list[str],
        year: int | None,
        limit: int,
        tool_context: ToolContext,
    ) -> dict:
        """Search report pages for this specialist's canonical KPI."""
        return search_kpi_report(kpi, query, phrases, year, limit, tool_context)

    return search_report


def _retrieval_budget_callback(kpi: str, candidate_tool_name: str):
    counts: dict[str, int] = {}
    limits = {
        candidate_tool_name: 1,
        "search_report": KPI_SPECIALIST_SEARCH_LIMIT,
        "read_report_pages": 2,
        "record_multi_kpi_progress": 3,
    }

    async def enforce_budget(
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict | None:
        del args
        limit = limits.get(tool.name)
        if limit is None:
            return None
        count = counts.get(tool.name, 0) + 1
        counts[tool.name] = count
        if count <= limit:
            return None
        if count == limit + 1:
            return {
                "status": "error",
                "retryable": False,
                "error": f"{kpi} specialist budget exhausted for {tool.name}",
                "next_action": (
                    "record ambiguous if relevant evidence is unresolved, otherwise record absent"
                ),
            }

        report = tool_context.state.get(REPORT_STATE_KEY, {})
        fiscal_year = report.get("year") if isinstance(report, dict) else None
        fallback = record_multi_kpi_progress(
            reporting_currency=None,
            units_note=None,
            kpis=[{"kpi": kpi, "fiscal_year": fiscal_year, "status": "ambiguous"}],
            notes=[],
            tool_context=tool_context,
        )
        if fallback.get("status") in {"success", "partial_success"}:
            tool_context.actions.skip_summarization = True
        return {
            **fallback,
            "fallback_reason": (
                f"{kpi} specialist repeated {tool.name} after its bounded source check"
            ),
        }

    return enforce_budget


def _finish_after_recording(kpi: str):
    async def finish_after_recording(
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
        tool_response: dict,
    ) -> dict | None:
        del tool, args
        work_record = tool_context.state.get(MULTI_KPI_WORK_RECORD_STATE_KEY, {})
        report = tool_context.state.get(REPORT_STATE_KEY, {})
        report_year = report.get("year") if isinstance(report, dict) else None
        rows = work_record.get("kpis", []) if isinstance(work_record, dict) else []
        recorded = next(
            (
                row
                for row in rows
                if isinstance(row, dict)
                and row.get("kpi") == kpi
                and row.get("fiscal_year") == report_year
            ),
            None,
        )
        if recorded is not None and tool_response.get("status") in {"success", "partial_success"}:
            tool_context.actions.skip_summarization = True
            return {
                "status": "success",
                "kpi": kpi,
                "fiscal_year": report_year,
                "coverage_status": recorded.get("status"),
                "value": recorded.get("value"),
                "page": recorded.get("page"),
                "coverage_count": tool_response.get("coverage_count"),
                "pending_count": tool_response.get("pending_count"),
            }
        return None

    return finish_after_recording


def _force_specialist_closure(kpi: str):
    turns = 0

    async def force_specialist_closure(
        callback_context: Any,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        nonlocal turns
        turns += 1
        if turns < KPI_SPECIALIST_MODEL_TURN_LIMIT:
            return None
        function_calls = llm_response.get_function_calls()
        if len(function_calls) == 1 and function_calls[0].name == "record_multi_kpi_progress":
            return None
        guarded = llm_response.model_copy(deep=True)
        report = callback_context.state.get(REPORT_STATE_KEY, {})
        fiscal_year = report.get("year") if isinstance(report, dict) else None
        guarded.content = genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(
                    function_call=genai_types.FunctionCall(
                        name="record_multi_kpi_progress",
                        args={
                            "reporting_currency": None,
                            "units_note": None,
                            "kpis": [
                                {
                                    "kpi": kpi,
                                    "fiscal_year": fiscal_year,
                                    "status": "ambiguous",
                                }
                            ],
                            "notes": [],
                        },
                    )
                )
            ],
        )
        return guarded

    return force_specialist_closure


def build_specialist_agent(
    *,
    kpi: str,
    description: str,
    instruction: str,
    max_output_tokens: int,
) -> Agent:
    """Attach module-owned KPI behavior to the common ADK runtime."""
    candidate_tool = _candidate_tool(kpi)
    return Agent(
        name=kpi_agent_name(kpi),
        model=ADK_MODEL,
        description=description,
        instruction=instruction,
        include_contents="none",
        tools=[
            candidate_tool,
            _search_tool(kpi),
            read_report_pages,
            _specialized_record_tool(kpi),
        ],
        before_tool_callback=_retrieval_budget_callback(kpi, candidate_tool.__name__),
        after_model_callback=_force_specialist_closure(kpi),
        after_tool_callback=_finish_after_recording(kpi),
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=max_output_tokens,
            tool_config=genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(
                    mode=genai_types.FunctionCallingConfigMode.ANY,
                )
            ),
        ),
    )
