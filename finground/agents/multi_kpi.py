"""Google ADK coordinator and application for LEDGER Multi-KPI extraction."""

from __future__ import annotations

import math
import re
from typing import Any

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps.app import App
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
from google.adk.tools import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

from finground.agents.common import ADK_MODEL
from finground.agents.kpi_specialists import (
    COMMON_TASK_AGENT_NAME,
    KPI_AGENT_FACTORIES,
    KPI_DISPATCH_TOOL_NAME,
    KPI_SPECIALIST_SEARCH_LIMIT,
    MULTI_KPI_COORDINATOR_NAME,
    KpiSpecialistTool,
)
from finground.context import filter_recorded_multi_kpi_context
from finground.kpis import KPI_ALIASES, KPI_DESCRIPTIONS, KPI_KEYS
from finground.tools import (
    MULTI_KPI_REQUESTED_STATE_KEY,
    MULTI_KPI_RESULT_STATE_KEY,
    finalize_multi_kpi_report,
    prepare_multi_kpi_report,
    query_multi_kpi_progress,
)

MULTI_KPI_APP_NAME = "finground_multi_kpi"
MULTI_KPI_PROMPT_VERSION = "independent-kpi-agents-v36"
MULTI_KPI_LLM_CALL_LIMIT = 200
MULTI_KPI_SEARCH_LIMIT = KPI_SPECIALIST_SEARCH_LIMIT
MULTI_KPI_PROGRESS_REMINDER_CALL = math.ceil(MULTI_KPI_LLM_CALL_LIMIT * 0.60)
MULTI_KPI_FINAL_WARNING_CALL = math.ceil(MULTI_KPI_LLM_CALL_LIMIT * 0.80)
MULTI_KPI_SUBMISSION_DEADLINE = MULTI_KPI_LLM_CALL_LIMIT
MULTI_KPI_CONTEXT_WINDOW_TOKENS = 128 * 1024
MULTI_KPI_MAX_OUTPUT_TOKENS = 4_096

KPI_CATALOGUE = "\n".join(
    f"- {key}: {description}" for key, description in KPI_DESCRIPTIONS.items()
)
KPI_AGENT_NAMES = tuple(f"extract_{kpi}" for kpi in KPI_KEYS)
MULTI_KPI_INSTRUCTION = f"""Coordinate a structured extraction workflow by delegating work to
context-isolated specialist tools. Do not inspect report text, choose evidence, calculate values,
or record KPI rows yourself.

WORKFLOW:
0. The authoritative requested KPI scope is in session state. A single-KPI or multi-KPI user
   request must process only that scope; the benchmark explicitly requests all 31.
1. Call {COMMON_TASK_AGENT_NAME} first and request report preparation. It indexes report-wide
   metadata and primary statements without adding full report pages to your context.
2. Delegate each still-pending KPI exactly once to its matching specialist tool. Every specialist
   owns one canonical KPI, validates its own evidence, and persists exactly one status:
   found, explicit_zero, absent, or ambiguous. Treat its returned result as authoritative.
3. Call {KPI_DISPATCH_TOOL_NAME} once with the complete pending KPI list and a compact request.
   The dispatcher runs each specialist in its own child session. Do not infer a KPI result from
   another specialist's output. Context isolation is intentional.
4. After all requested specialists return, call {COMMON_TASK_AGENT_NAME} for a coverage check.
   If it reports pending KPIs, call only those named specialist tools, passing the validation
   feedback in the request.
5. When every requested KPI is covered, call {COMMON_TASK_AGENT_NAME} for final submission.
   Finish only when that tool returns status=success and completion_status=complete.

Use exactly one tool call per response and no prose. Pass pending KPIs in canonical order:
{", ".join(KPI_KEYS)}
"""


def resolve_requested_kpis(text: str) -> list[str]:
    """Resolve explicit canonical names/aliases; broad requests mean all KPIs."""
    raw = text.casefold()
    canonical_matches = [
        kpi
        for kpi in KPI_KEYS
        if re.search(rf"(?<![a-z0-9_]){re.escape(kpi)}(?![a-z0-9_])", raw)
    ]
    if canonical_matches:
        return canonical_matches
    normalized = re.sub(r"[_\-/]+", " ", text.casefold())
    if any(marker in normalized for marker in ("all kpi", "all supported", "31 kpi")):
        return list(KPI_KEYS)
    matched = []
    for kpi in KPI_KEYS:
        labels = (kpi.replace("_", " "), *KPI_ALIASES[kpi])
        if any(label.casefold() in normalized for label in labels):
            matched.append(kpi)
    return matched or list(KPI_KEYS)


async def _set_requested_kpi_scope(callback_context: CallbackContext) -> None:
    requested = callback_context.state.get(MULTI_KPI_REQUESTED_STATE_KEY)
    if isinstance(requested, list) and requested:
        return
    events = callback_context._invocation_context.session.events
    for event in reversed(events):
        content = event.content
        if content is None or content.role != "user" or not content.parts:
            continue
        text = "\n".join(part.text for part in content.parts if part.text)
        callback_context.state[MULTI_KPI_REQUESTED_STATE_KEY] = resolve_requested_kpis(text)
        return
    if MULTI_KPI_REQUESTED_STATE_KEY not in callback_context.state:
        callback_context.state[MULTI_KPI_REQUESTED_STATE_KEY] = list(KPI_KEYS)


COMMON_TASK_INSTRUCTION = """You handle only report-wide workflow tasks for the Multi-KPI
coordinator. You run in a fresh isolated context for each request. Never find, judge, calculate,
or record an individual KPI value.

Choose exactly one tool from the coordinator's request:
- Preparation or initialization: call prepare_multi_kpi_report.
- Coverage, status, pending work, or audit: call query_multi_kpi_progress with view="kpis".
- Finalize or submit: call finalize_multi_kpi_report.

Return the selected tool result immediately. Preparation must precede KPI delegation. Finalization
is valid only after every KPI in the requested session scope has a persisted status."""


async def _return_common_tool_result(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict,
) -> dict | None:
    del tool, args, tool_response
    tool_context.actions.skip_summarization = True
    return None


async def _finish_after_submission(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: Any,
) -> dict | None:
    del args, tool_response
    if tool.name == COMMON_TASK_AGENT_NAME and isinstance(
        tool_context.state.get(MULTI_KPI_RESULT_STATE_KEY), dict
    ):
        tool_context.actions.skip_summarization = True
    return None


def create_common_task_agent() -> Agent:
    """Create the context-isolated report-wide workflow specialist."""
    return Agent(
        name=COMMON_TASK_AGENT_NAME,
        model=ADK_MODEL,
        description=(
            "Prepares report-wide metadata and primary-statement indexes, audits 31-KPI coverage, "
            "and submits the completed extraction; never decides an individual KPI value."
        ),
        instruction=COMMON_TASK_INSTRUCTION,
        include_contents="none",
        tools=[
            prepare_multi_kpi_report,
            query_multi_kpi_progress,
            finalize_multi_kpi_report,
        ],
        after_tool_callback=_return_common_tool_result,
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=MULTI_KPI_MAX_OUTPUT_TOKENS,
            tool_config=genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(
                    mode=genai_types.FunctionCallingConfigMode.ANY,
                )
            ),
        ),
    )


def create_kpi_agent_tools() -> list[KpiSpecialistTool]:
    """Create one compact dispatcher whose child agents remain KPI-specific."""
    return [KpiSpecialistTool(max_output_tokens=MULTI_KPI_MAX_OUTPUT_TOKENS)]


def create_multi_kpi_agent() -> Agent:
    """Create the generic coordinator with 31 KPI and one common sub-agent."""
    common_task_tool = AgentTool(create_common_task_agent(), include_plugins=True)
    return Agent(
        name=MULTI_KPI_COORDINATOR_NAME,
        model=ADK_MODEL,
        description=(
            "Coordinates context-isolated specialists to complete a structured extraction workflow."
        ),
        instruction=MULTI_KPI_INSTRUCTION,
        before_agent_callback=_set_requested_kpi_scope,
        tools=[common_task_tool, *create_kpi_agent_tools()],
        after_tool_callback=_finish_after_submission,
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=MULTI_KPI_MAX_OUTPUT_TOKENS,
            tool_config=genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(
                    mode=genai_types.FunctionCallingConfigMode.ANY,
                )
            ),
        ),
    )


def create_multi_kpi_app(*, plugins: list[BasePlugin] | None = None) -> App:
    """Create the ADK app with isolated child sessions and deterministic filtering."""
    return App(
        name=MULTI_KPI_APP_NAME,
        root_agent=create_multi_kpi_agent(),
        plugins=[
            ContextFilterPlugin(custom_filter=filter_recorded_multi_kpi_context),
            *(plugins or []),
        ],
    )


if len(KPI_AGENT_FACTORIES) != len(KPI_KEYS):
    raise RuntimeError("Multi-KPI coordinator requires exactly one specialist per KPI")
